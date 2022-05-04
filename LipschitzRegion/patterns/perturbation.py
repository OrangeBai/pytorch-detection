from settings.cifar_settings import *
from models.mini.model import Model
from dataloaders.cifar import *
from patterns.tools import *
import time
from core.utils import save_result

if __name__ == '__main__':
    args = set_up_training(False)
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_loader, test_loader = get_loaders(args.data_dir, 1, args.dataset)
    global_recorder = []
    t = time.time()
    batch_size = 30
    e = 1
    recorder = []
    for _, (img, label) in enumerate(test_loader):
        model = Model(args, None)
        post_activation, handles = add_hook(model, torch.nn.ReLU)
        model.load_model(args.model_dir)
        for idx in range(batch_size):
            perturbation = add_noise(img.squeeze(), 101, epsilon=e)
            model.model(perturbation.cuda())
            if idx + 1 >= batch_size:
                break
        for handle in handles:
            handle.remove()
        batch_pattern = get_batch_pattern(post_activation, [0])

        region_diff = []
        for layer, patterns in batch_pattern.items():
            a = []
            diff = np.zeros(patterns.shape[1], dtype=bool)
            for idx, instance_pattern in enumerate(patterns[1:]):
                if idx % 100 == 0:
                    a.append(diff.sum() / len(diff))
                else:
                    diff += patterns[0] != instance_pattern
            region_diff += [a]
        recorder += [region_diff]
        print(time.time() - t)
        t = time.time()
        if len(recorder) >= 32:
            break
    save_result(np.array(recorder), args.model_dir, '1_portion')
    print(1)
