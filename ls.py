import os

def count_samples(dataset_dir):
    total_samples = 0
    for person in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person)
        if not os.path.isdir(person_dir):
            continue
        for eye in ['left_eye', 'right_eye']:
            eye_dir = os.path.join(person_dir, eye)
            if not os.path.exists(eye_dir):
                continue
            num_files = len(os.listdir(eye_dir))
            print(f"{person}/{eye}: {num_files} 张")
            total_samples += num_files
    print(f"总样本数: {total_samples}")

print("训练集样本分布：")
count_samples('iris_dataset/train')

print("\n测试集样本分布：")
count_samples('iris_dataset/test')
