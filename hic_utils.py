import os
import numpy as np
import scipy
import scipy.sparse
from config import *
import hicstraw
from itertools import combinations_with_replacement
import json
from tqdm import tqdm
import re

def chr2id(chr):
    # convert chrN form chromosome to a int id.
    try:
        chr = int(chr)
    except:
        pass

    if isinstance(chr, int):
        return chr
    
    if not isinstance(chr, str):
        print(chr)
        raise NotImplementedError   

    if chr[:3] == 'chr':
        chr = chr[3:]
    if chr == 'X': return 100
    if chr == 'Y': return 101
    
    return int(chr)

def id2chr(id): 
    # convert a int id to chrN form.

    try:
        id = int(id)
    except:
        pass

    if isinstance(id, int):
        if id == 100:
            ch = 'X'
        elif id == 101:
            ch = 'Y'
        else:
            ch = str(id)
    elif isinstance(id, str):
        ch = id 
    else:
        print(id)
        raise NotImplementedError   

    if ch[:3] != 'chr':
        ch = "chr"+ch

    return ch

def find_chr_in_set(chr, chr_set):
    # first try standard format: 'chrN'/'N'
    c = id2chr(chr)
    if c in chr_set:
        return c
    elif c[3:] in chr_set:
        return c[3:]

    if isinstance(chr, str):
        return chr
    else:
        return list(chr_set)[chr]
    
    raise NotImplementedError

def extract_property(hic_file, max_chromosomes = 200):
    pfile = os.path.join(hic_property_dir, os.path.basename(hic_file) + '.property')

    hic = hicstraw.HiCFile(hic_file)
    chromosomes = hic.getChromosomes()
    filtered_chromosomes = [ch for ch in chromosomes if ch.name != 'ALL' and ch.name != 'All']

    # Assemblies with many unplaced scaffolds give millions of chromosome pairs to scan, which
    # exhausts hicstraw's file handles ("cannot be opened for reading"). Only the read-count to
    # genome-length ratio is used, so restrict both to the largest sequences in that case.
    if len(filtered_chromosomes) > max_chromosomes:
        keep = set(ch.name for ch in sorted(filtered_chromosomes, key=lambda ch: ch.length, reverse=True)[:max_chromosomes])
        filtered_chromosomes = [ch for ch in filtered_chromosomes if ch.name in keep]   # keep file order
        print(f'{hic_file}: {len(chromosomes)} sequences, computing read coverage on the {max_chromosomes} largest only')

    try:
        with open(pfile) as f:
            properties = json.load(f)
    except:
        properties = {}

    updated = False

    if 'mapped_reads' not in properties or properties['mapped_reads'] <= 0:

        resolutions = hic.getResolutions()
        base_resolution = max(resolutions)
        mapped_reads = 0
        intra_mapped_reads = 0
        for ch0, ch1 in tqdm(combinations_with_replacement(filtered_chromosomes, 2)):
            n0, n1 = ch0.name, ch1.name
            
            if n0 == 'ALL' or n1 == 'ALL': continue
            if n0 == 'All' or n1 == 'All': continue

            L0, L1 = ch0.length, ch1.length
            try:
                mzd = hic.getMatrixZoomData(n0, n1, "observed", 'NONE', "BP", base_resolution)
                records = mzd.getRecords(0, L0, 0, L1)
            except Exception:
                # the pair has no contact map in this file
                continue
            if len(records) <= 0 : continue

            s = np.sum([record.counts for record in records])

            mapped_reads += s
            if n0 == n1:
                intra_mapped_reads += s

        properties['mapped_reads'] = mapped_reads
        properties['intra_mapped_reads'] = intra_mapped_reads

        updated = True
    
    if 'total_length' not in properties or properties['total_length'] <= 0:
        total_length = 0
        for ch in filtered_chromosomes:
            total_length += ch.length

        properties['total_length'] = total_length

        updated = True

    if updated:
        print(properties)

        with open(pfile, 'w') as f:
            json.dump(properties, f, indent=2)

    return properties['mapped_reads'] / properties['total_length'] 
    
def save_hic(matrices, reference, file_name, resolution=None, first = True, finished = True, norm = True):

    temp_name = file_name + '.temp'

    with open(temp_name, 'w' if first else 'a') as temp_file:
        for k, mat in matrices.items():
            ch0, ch1 = k

            if ch0 != ch1: inter = True

            for x, y, score in zip(mat.row, mat.col, mat.data):
                if isinstance(score, (int, np.integer)):
                    temp_file.write(f'{0} {ch0} {x} {0} {0} {ch1} {y} {1} {score}\n')
                else:
                    temp_file.write(f'{0} {ch0} {x} {0} {0} {ch1} {y} {1} {score:.4f}\n')

    if finished:
        command = f'java -Xmx256g -jar {juicer_tool_path} pre -j 8 {temp_name} {file_name} {reference}'
        if resolution is not None:
            command = command + f' -r {resolution}'
        else:
            command = command + f' -r 2000,4000,8000,10000'
        if not norm :
            command = command + ' -n'

        print(command)
        code = os.system(command)
        
        if code == 0:
            os.remove(temp_name)

def find_chr_in_hic(c, hic):
    for ch in hic.getChromosomes():
        if ch.name == c:
            return ch
    return None

def records2npy(records):
    
    X    = np.array([record.binX   for record in records])
    Y    = np.array([record.binY   for record in records])
    data = np.array([record.counts for record in records])

    return X, Y, data

def hic2upper(X, Y, data, length) -> scipy.sparse.csr_matrix:
    diag = Y - X
    idx = np.where((diag >= 0))
    idxRow = diag[idx]
    idxCol = Y[idx] - idxRow
    ans = scipy.sparse.csr_matrix((data[idx], (idxRow, idxCol)),
                        shape=(length, length), dtype=data.dtype)
    ans.eliminate_zeros()
    return ans

def hic2sparse(X, Y, data, length) -> scipy.sparse.csr_matrix:
    ans = scipy.sparse.csr_matrix((data, (X, Y)),
                        shape=(length, length), dtype=data.dtype)
    ans.eliminate_zeros()
    return ans

def extend(X:np.ndarray, Y:np.ndarray, data:np.ndarray, gr:int, br:int):    
    assert br % gr == 0
    finalX, finalY, finaldata = [], [], []

    for x, y, d in sorted(zip(X.tolist(), Y.tolist(), data.tolist())):
        if d==0: continue
        v = br//gr
        new_d = d/v/v
        for i in range(0, br, gr):
            for j in range(0, br, gr):
                finalX.append(x+i)
                finalY.append(y+j)
                finaldata.append(new_d)
    
    return np.array(finalX), np.array(finalY), np.array(finaldata)

def pool(X:np.ndarray, Y:np.ndarray, data:np.ndarray, r:int):    
    new_X = X // r * r
    new_Y = Y // r * r

    cur_X, cur_Y, cur_data  = 0, 0, 0
    finalX, finalY, finaldata = [], [], []

    for X, Y, data in sorted(zip(new_X.tolist(), new_Y.tolist(), data.tolist())):
        if X==cur_X and Y==cur_Y:
            cur_data += data
        elif data>0:
            finalX.append(cur_X)
            finalY.append(cur_Y)
            finaldata.append(cur_data)

            cur_X = X
            cur_Y = Y
            cur_data = data
    
    if data>0:
        finalX.append(cur_X)
        finalY.append(cur_Y)
        finaldata.append(cur_data)

    return np.array(finalX), np.array(finalY), np.array(finaldata)

def downsample_hic(hic_file, downsample, species, downsample_target = None, seed=0, resolution=2000, chromosome_filter=['ALL', 'All']):
    outputs = os.path.basename(hic_file).split('.')
    outputs = outputs[:-1] + [f'{downsample}' if downsample_target is None else f'h{int(downsample_target)}', f'seed{seed}', outputs[-1]]
    output = '.'.join(outputs)

    output = os.path.join(hic_downsample_dir if species in splits else dnazoo_downsample_dir, output)

    if os.path.isfile(output):
        return output

    if downsample_target is not None:
        # downsample_target in hg38
        downsample = extract_property(hic_file) * human_total_length / downsample_target
    
    print(f'Downsampling {output} with ratio {downsample} and seed {seed}')

    hic = hicstraw.HiCFile(hic_file)
    chromosomes = hic.getChromosomes()
    resolutions = hic.getResolutions()
    base_resolution = min(resolutions)
    if resolution is not None:
        for r in resolutions:
            if resolution % r == 0 and r > base_resolution:
                base_resolution = r

        def gcd(a,b):
            return a if b==0 else gcd(b, a%b)
        g_res = gcd(base_resolution, resolution)
        if g_res != base_resolution:
            print(f'No existing resolution could divide {resolution}. Using avg of {base_resolution} to {g_res} instead.')

    datas = {}
    
    for ch0, ch1 in combinations_with_replacement(chromosomes, 2):
        n0, n1 = ch0.name, ch1.name
        if n0 in chromosome_filter or n1 in chromosome_filter: continue
        print(n0, n1)

        mzd = hic.getMatrixZoomData(n0, n1, "observed", 'NONE', "BP", base_resolution)

        L0, L1 = ch0.length, ch1.length

        records = mzd.getRecords(0, L0, 0, L1)

        X, Y, data = records2npy(records)
        print(f'Read {len(data)} records with {np.sum(data)} reads.')

        data = np.random.binomial(data.astype(int), 1/downsample)

        if resolution is not None:
            if g_res != base_resolution:
                X, Y, data = extend(X, Y, data, g_res, base_resolution)
            X, Y, data = pool(X, Y, data, resolution)

        if len(data) == 0: continue

        sp_mat = scipy.sparse.coo_matrix((data, (X, Y)))
        sp_mat.eliminate_zeros()
        print(f'Downsampled. {sp_mat.getnnz()} records remains, with {np.sum(sp_mat.data)} reads.')
        
        if species in references:
            n0, n1 = id2chr(n0), id2chr(n1)
        
        datas[(n0, n1)] = sp_mat

    print(f'Saving the data to {output}')
    if species in references:
        reference = references[species]
    else:
        reference = output.replace('.hic', '.chrom.size')
        with open(reference, 'w') as f:
            for ch in chromosomes:
                if ch.name in chromosome_filter:continue
                f.write(f"{ch.name}\t{ch.length}\n")

    save_hic(datas, reference, output, first=True, finished=True)

    return output