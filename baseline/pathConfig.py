import os

if os.path.exists('D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'):
    base_dir = 'D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'
else:
    base_dir = '/media/dell/disk/jk/Retrieval'

res_dir = base_dir + '/baseline/res'

def get_base_path():
    return os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print(get_base_path())