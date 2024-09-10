def add_dict():
    from konlpy.tag import Okt
    import os
    import shutil
    import jpype

    okt = Okt()

    # print(okt.pos("순대국 먹고 싶다."))
    # print(okt.pos("순댓국 먹고 싶다."))
    # print(okt.pos("패스트파이브에서 일을 합니다."))
    # print(okt.pos("아이오아이는 정말 이뻐요."))

    directory = '/mnt/c/Users/USER/Desktop/nam/hai/HAI-Check/venv/lib/python3.10/site-packages/konlpy/java'
    os.chdir(directory)
    os.getcwd() 

    # !jar xvf open-korean-text-2.1.0.jar
    import subprocess

    # jar 파일 추출
    subprocess.run(['jar', 'xvf', 'open-korean-text-2.1.0.jar'], check=True)
    # data 확인
    with open(os.path.join(directory, "org/openkoreantext/processor/util/noun/names.txt")) as f:
        data = f.read()
        
    print(data)

    data += '꽭뛼쨿쀎\n'

    # 사전 저장
    with open(os.path.join(directory, "org/openkoreantext/processor/util/noun/names.txt"), 'w') as f:
        f.write(data)
        
    with open(os.path.join(directory, "org/openkoreantext/processor/util/noun/names.txt")) as f:
        data = f.read()
    print(data)

    # !jar cvf open-korean-text-2.1.0.jar org
    # !rm -r org
    subprocess.run(['jar', 'cvf', 'open-korean-text-2.1.0.jar', 'org'], cwd=directory, check=True)
    # shutil.rmtree(f'{directory}/org')
    
if __name__ == "__main__":
    add_dict()