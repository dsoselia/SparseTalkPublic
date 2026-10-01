from transformers import AutoConfig
import torch.nn as nn


def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if "llava" in config and "llava" not in cfg.model_type:
        assert cfg.model_type == "llama"
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = "LlavaLlamaForCausalLM"
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)

class CrossAttentionFusion(nn.Module):
    def __init__(self, config): 
        super().__init__()
        self.config = config
        self.feature_dim = config.hidden_size
        self.querry = nn.Linear(self.feature_dim, self.feature_dim//8)
        self.key = nn.Linear(self.feature_dim, self.feature_dim//8)
        self.value = nn.Linear(self.feature_dim, self.feature_dim//8)
        self.attention = nn.MultiheadAttention(self.feature_dim//8, 1)

        self.decode = nn.Linear(self.feature_dim//8, self.feature_dim)


    def forward(self, feature_3d, feature_2d):
        feature_3d_norm = feature_3d / (feature_3d.norm(dim=-1, keepdim=True) + 1e-5)
        feature_2d_norm = feature_2d / (feature_2d.norm(dim=-1, keepdim=True) + 1e-5)
        q = self.querry(feature_3d)
        k = self.key(feature_2d)
        # v = self.value_2d(feature_2d)
        v = self.value(feature_2d)

        # output, _ = self.attention(q, k, v)
        output, _ = self.attention(q, k, v)

        output = self.decode(output) + feature_3d
        # output, _ = self.attention(k, q, v)

        return output
        # return output


