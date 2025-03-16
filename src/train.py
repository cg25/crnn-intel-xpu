import torch
import yaml
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from core.crnn import CRNN
from core.dataset import CRNNDataset
from pathlib import Path


def main(config_path="configs/train.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['device'])
    torch.xpu.empty_cache()

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        targets = torch.cat([item[1] for item in batch])
        return images.to(device), targets.to(device)

    dataset = CRNNDataset(
        data_root=cfg['data_root'],
        img_h=cfg['img_h'],
        charset_path=cfg['charset_path']
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    model = CRNN(
        img_h=cfg['img_h'],
        num_classes=dataset.num_classes
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    criterion = CTCLoss(blank=0)

    Path(cfg['checkpoint_dir']).mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg['epochs']):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, targets) in enumerate(dataloader):
            optimizer.zero_grad()

            outputs = model(images)  # [seq_len, batch, num_classes]

            input_lengths = torch.full(
                size=(outputs.size(1),),  # batch size
                fill_value=outputs.size(0),  # seq_len
                dtype=torch.long,
                device=device
            )

            target_lengths = torch.full(
                size=(images.size(0),),  # batch size
                fill_value=len(dataset.img_paths[0].stem),
                dtype=torch.long,
                device=device
            )

            loss = criterion(
                outputs.log_softmax(2),  # [T, N, C]
                targets,  # [sum(target_lengths)]
                input_lengths,
                target_lengths
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{cfg['epochs']}] "
                      f"Batch [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        checkpoint_path = Path(cfg['checkpoint_dir']) / "latest.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': avg_loss,
        }, str(checkpoint_path))

        print(f"Epoch [{epoch + 1}/{cfg['epochs']}] checkpoint overwritten")

        print(f"Epoch [{epoch + 1}/{cfg['epochs']}] "
              f"Average Loss: {avg_loss:.4f} "
              f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()