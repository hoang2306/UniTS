import torch
import sys
import os

# Put project dir on path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.UniTS import Model

class Args:
    d_model = 64
    n_heads = 4
    e_layers = 1
    dropout = 0.1
    prompt_num = 3
    patch_len = 8
    stride = 8
    d_text = 1024
    text_fusion_type = 'add_patch'

def run_test():
    print("Initializing test...")
    args = Args()
    
    # configs_list: list of [task_data_name, config_dict]
    # Anomaly detection configs: seq_len = 24, enc_in = 2, dec_in = 2, c_out = 2
    configs_list = [
        [
            "Energy_AD",
            {
                "task_name": "anomaly_detection",
                "dataset": "Energy",
                "seq_len": 24,
                "enc_in": 2,
                "dec_in": 2,
                "c_out": 2
            }
        ]
    ]
    
    fusion_types = ['add_patch', 'add_prompt', 'concat_prompt']
    
    for fusion_type in fusion_types:
        print(f"\n--- Testing with fusion_type: {fusion_type} ---")
        args.text_fusion_type = fusion_type
        
        # Instantiate model
        model = Model(args, configs_list)
        
        # Make mock inputs
        # batch_size=4, L=24, V=2
        B = 4
        L = 24
        V = 2
        
        x = torch.randn(B, L, V)
        text_embeddings = torch.randn(B, L, args.d_text)
        
        # Forward pass
        print("Running forward pass...")
        outputs = model(x, None, task_id=0, task_name='anomaly_detection', text_embeddings=text_embeddings)
        print(f"Output shape: {outputs.shape} (Expected: {[B, L, V]})")
        assert list(outputs.shape) == [B, L, V], f"Output shape mismatch! Got {outputs.shape}"
        
        # Compute dummy loss and backward pass
        print("Running backward pass...")
        loss = outputs.pow(2).mean()
        loss.backward()
        
        # Check text adapter gradients
        print("Checking gradients...")
        has_grad = False
        for name, param in model.named_parameters():
            if 'text_adapter' in name:
                if param.grad is not None:
                    has_grad = True
                    print(f"  param '{name}' has grad: True")
                else:
                    print(f"  param '{name}' has grad: False")
        
        assert has_grad, "Text adapter parameters did not receive any gradients!"
        print(f"Fusion type {fusion_type} passed successfully!")

    print("\nAll integration verification tests passed successfully!")

if __name__ == '__main__':
    run_test()
