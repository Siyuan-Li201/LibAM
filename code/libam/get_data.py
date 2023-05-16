import os, json
import subprocess

def execute_cmd(cmd, timeout=300):
    '''
    execute system command
    :param cmd:
    :return:
    '''
    try:
        p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'errmsg': 'Timeout'
        }
    else:
        msg = p.stdout.decode()

        if p.returncode == 0:
            return {
                'success': True,
                'msg': msg
            }
        else:
            return {
                'success': False,
                'errmsg': msg
            }



def get_candidate():
    dataset_path = "/data/lisiyuan/work2023/data/libae/dataset-elf-exec"

    target_path = "/data/lisiyuan/work2023/result/libae/libae_v2.0/iot_data/1_binary/candidate"

    arch = ["X64"]
    # tar_arch = ["arm-32", "x86-32", "x86-64"]

    opti = ["O2"]


    for arch_item in os.listdir(os.path.join(dataset_path)):
        if arch_item in arch:
            for opti_item in os.listdir(os.path.join(dataset_path, arch_item)):
                if opti_item in opti:
                    for project_item in os.listdir(os.path.join(dataset_path, arch_item, opti_item)):
                        for bianry_item in os.listdir(os.path.join(dataset_path, arch_item, opti_item, project_item)):
                            os.system("cp " + os.path.join(dataset_path, arch_item, opti_item, project_item, bianry_item) + " " + os.path.join(target_path, project_item+"___"+bianry_item))


def get_target():
    dataset_path = "/data/lisiyuan/extract_firmware/extract_firmware/ASUS/_Rescue_DSL_AC68U_30043762048.zip.extracted"
    
    model_name = "ASUS___AC68U_30043762048"

    target_path = "/data/lisiyuan/work2023/result/libae/libae_v2.0/iot_data/1_binary/target"

    # arch = ["X64"]
    # # tar_arch = ["arm-32", "x86-32", "x86-64"]

    # opti = ["O2"]

        
    # for model_dir in os.listdir(os.path.join(dataset_path)):
    #     if os.path.isdir(os.path.join(dataset_path)):
    for root, dirs, files in os.walk(os.path.join(dataset_path), topdown=False):
        for name in files:
            # print(os.path.join(root, name))
            command = f"file {os.path.join(root, name)}"
            execute_res = execute_cmd(command)
            if execute_res['success']:
                file_res = execute_res['msg']
                if " ELF " in file_res and "executable" in file_res:
                    print("cp " + os.path.join(root, name) + " " + os.path.join(target_path, model_name+"___"+name))
                    os.system("cp " + os.path.join(root, name) + " " + os.path.join(target_path, model_name+"___"+name))
    #     if arch_item in arch:
    #         for opti_item in os.listdir(os.path.join(dataset_path, arch_item)):
    #             if opti_item in opti:
    #                 for project_item in os.listdir(os.path.join(dataset_path, arch_item, opti_item)):
    #                     for bianry_item in os.listdir(os.path.join(dataset_path, arch_item, opti_item, project_item)):
    #                         os.system("cp " + os.path.join(dataset_path, arch_item, opti_item, project_item, bianry_item) + " " + os.path.join(target_path, project_item+"___"+bianry_item))
get_target()