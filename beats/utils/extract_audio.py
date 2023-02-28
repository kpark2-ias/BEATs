import sys
from glob import glob
from tempfile import mkdtemp
import os
import subprocess 



def extract_audio(indir,outdir='wavs', num=None,**kwargs):


    # tdir = mkdtemp(dir=os.getcwd())
    os.makedirs(outdir,exist_ok=True)
    print(indir)
    processed = []
    for fnum,inpath in enumerate(glob(f'{indir}/*.mp4')):
        print(17,num,fnum)
        if num is not None:
            if fnum >= num:
                break
        processed.append(inpath)
        outpath = f'{outdir}/'+os.path.basename(inpath).split('.')[0]+'.wav'

        command1 = f'ffmpeg -y -i {inpath} -vn -ar 16000 -ac 1 {outpath}'
        print(command1)
        subprocess.check_call(command1, shell=True)

    if not processed:
        outdir = indir

    outdir_chunked = f'{outdir}_chunked'
    os.makedirs(outdir_chunked,exist_ok=True)




    command2 = (
            f'find {outdir} -name "*.wav" | '
            'parallel '
            '\'ffmpeg -i {} -f segment -segment_time 5 -c copy '
            f'\"{outdir_chunked}\"/'
            '{/}%05d.wav\''
            )

    print(command2)
    subprocess.check_call(command2,shell=True)
    return outdir_chunked


if __name__ == '__main__':
    extract_audio(sys.argv[1])

