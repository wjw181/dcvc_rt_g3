#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage Transition Check Script
Check if current training stage is ready for next stage
"""

import re
import sys
import os
from typing import List, Tuple, Dict

def parse_log_file(log_file: str) -> Tuple[List[Dict], List[Dict]]:
    """Parse log file and extract training and test results"""
    if not os.path.exists(log_file):
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Extract test results
    test_results = []
    for line in lines:
        if 'Test epoch' in line:
            # Test epoch 21: Loss: 2.184 | PSNR: 19.609 | BPP: 0.1253
            epoch_match = re.search(r'Test epoch (\d+):', line)
            psnr_match = re.search(r'PSNR: ([\d.]+)', line)
            loss_match = re.search(r'Loss: ([\d.]+)', line)
            bpp_match = re.search(r'BPP: ([\d.]+)', line)
            
            if epoch_match and psnr_match and loss_match and bpp_match:
                test_results.append({
                    'epoch': int(epoch_match.group(1)),
                    'loss': float(loss_match.group(1)),
                    'psnr': float(psnr_match.group(1)),
                    'bpp': float(bpp_match.group(1))
                })
    
    # Extract training results (last iteration of each epoch)
    train_results = []
    current_epoch = -1
    last_train_line = {}
    
    for line in lines:
        # Detect epoch start
        epoch_match = re.search(r'====== Epoch (\d+) ======', line)
        if epoch_match:
            # Save last epoch's training result
            if last_train_line and current_epoch >= 0:
                train_results.append(last_train_line)
            current_epoch = int(epoch_match.group(1))
            last_train_line = {'epoch': current_epoch}
        
        # Extract training metrics (usually every 500 iterations)
        if '[' in line and 'Loss:' in line and 'PSNR:' in line:
            # [20000/20192] | Loss: 68.506 | PSNR: 22.373 | BPP: 0.4012 | LPIPS: 0.4777
            iter_match = re.search(r'\[(\d+)/\d+\]', line)
            loss_match = re.search(r'Loss: ([\d.]+)', line)
            psnr_match = re.search(r'PSNR: ([\d.]+)', line)
            bpp_match = re.search(r'BPP: ([\d.]+)', line)
            lpips_match = re.search(r'LPIPS: ([\d.]+)', line)
            latent_mse_match = re.search(r'LatentMSE: ([\d.]+)', line)
            
            if iter_match and loss_match and psnr_match:
                iteration = int(iter_match.group(1))
                # Only keep last record of each epoch (close to 20000/20192)
                if iteration >= 19000:
                    last_train_line = {
                        'epoch': current_epoch,
                        'loss': float(loss_match.group(1)),
                        'psnr': float(psnr_match.group(1)),
                        'bpp': float(bpp_match.group(1)) if bpp_match else 0,
                        'lpips': float(lpips_match.group(1)) if lpips_match else 0,
                        'latent_mse': float(latent_mse_match.group(1)) if latent_mse_match else 0
                    }
    
    # Add last epoch's training result
    if last_train_line and current_epoch >= 0:
        train_results.append(last_train_line)
    
    return train_results, test_results

def check_stage1_transition(train_results: List[Dict], test_results: List[Dict]) -> bool:
    """Check if Stage 1 is ready to transition to Stage 2"""
    
    print(f"\n{'='*70}")
    print("Stage 1 -> Stage 2 Transition Check")
    print(f"{'='*70}\n")
    
    if len(test_results) < 5:
        print("Error: Insufficient test data (need at least 5 epochs)")
        return False
    
    if len(train_results) < 5:
        print("Error: Insufficient training data (need at least 5 epochs)")
        return False
    
    # Recent 5 epochs
    recent_test = test_results[-5:]
    recent_train = [r for r in train_results[-5:] if 'psnr' in r and r['psnr'] > 0]
    
    if len(recent_train) < 3:
        print(f"Warning: Only {len(recent_train)} valid training epochs found in recent data")
        print("Using all available training data...")
        recent_train = [r for r in train_results if 'psnr' in r and r['psnr'] > 0]
    
    if not recent_train:
        print("Error: No valid training data with PSNR found")
        return False
    
    # Calculate statistics
    test_psnr_avg = sum(r['psnr'] for r in recent_test) / len(recent_test)
    test_psnr_max = max(r['psnr'] for r in recent_test)
    test_psnr_min = min(r['psnr'] for r in recent_test)
    test_bpp_avg = sum(r['bpp'] for r in recent_test) / len(recent_test)
    
    train_psnr_avg = sum(r['psnr'] for r in recent_train) / len(recent_train)
    train_psnr_max = max(r['psnr'] for r in recent_train)
    train_loss_avg = sum(r['loss'] for r in recent_train) / len(recent_train)
    train_latent_mse_avg = sum(r.get('latent_mse', 0) for r in recent_train) / len(recent_train)
    train_lpips_avg = sum(r.get('lpips', 0) for r in recent_train) / len(recent_train)
    
    # Display current status
    print("Current Training Status (Last 5 Epochs):")
    print(f"  Epoch Range: {recent_test[0]['epoch']} - {recent_test[-1]['epoch']}")
    print()
    print("  Test Metrics:")
    print(f"    Avg PSNR:      {test_psnr_avg:.2f} dB")
    print(f"    Max PSNR:      {test_psnr_max:.2f} dB")
    print(f"    PSNR Variance: {test_psnr_max - test_psnr_min:.2f} dB")
    print(f"    Avg BPP:       {test_bpp_avg:.4f}")
    print()
    print("  Train Metrics:")
    print(f"    Avg PSNR:      {train_psnr_avg:.2f} dB")
    print(f"    Max PSNR:      {train_psnr_max:.2f} dB")
    print(f"    Avg Loss:      {train_loss_avg:.2f}")
    print(f"    Avg LatentMSE: {train_latent_mse_avg:.4f}")
    print(f"    Avg LPIPS:     {train_lpips_avg:.4f}")
    print()
    
    # Check transition conditions
    print(f"{'='*70}")
    print("Transition Condition Checks:")
    print(f"{'='*70}\n")
    
    checks = []
    
    # Required conditions
    check_test_psnr_min = test_psnr_avg >= 25.0
    check_test_psnr_target = test_psnr_avg >= 26.0
    checks.append(("Test PSNR >= 25 dB (REQUIRED)", test_psnr_avg, 25.0, check_test_psnr_min, True))
    checks.append(("Test PSNR >= 26 dB (TARGET)", test_psnr_avg, 26.0, check_test_psnr_target, False))
    
    check_train_psnr_min = train_psnr_avg >= 26.0
    check_train_psnr_target = train_psnr_avg >= 27.0
    checks.append(("Train PSNR >= 26 dB (REQUIRED)", train_psnr_avg, 26.0, check_train_psnr_min, True))
    checks.append(("Train PSNR >= 27 dB (TARGET)", train_psnr_avg, 27.0, check_train_psnr_target, False))
    
    check_lpips = train_lpips_avg <= 0.50
    check_lpips_target = train_lpips_avg <= 0.45
    checks.append(("Train LPIPS <= 0.50 (REQUIRED)", train_lpips_avg, 0.50, check_lpips, True))
    checks.append(("Train LPIPS <= 0.45 (TARGET)", train_lpips_avg, 0.45, check_lpips_target, False))
    
    check_latent_mse = train_latent_mse_avg < 0.15
    check_latent_mse_target = train_latent_mse_avg < 0.12
    checks.append(("LatentMSE < 0.15 (REQUIRED)", train_latent_mse_avg, 0.15, check_latent_mse, True))
    checks.append(("LatentMSE < 0.12 (TARGET)", train_latent_mse_avg, 0.12, check_latent_mse_target, False))
    
    check_stability = (test_psnr_max - test_psnr_min) < 1.5
    check_stability_target = (test_psnr_max - test_psnr_min) < 1.0
    checks.append(("PSNR Stability (var<1.5dB)", test_psnr_max - test_psnr_min, 1.5, check_stability, True))
    checks.append(("PSNR Stability (var<1.0dB)", test_psnr_max - test_psnr_min, 1.0, check_stability_target, False))
    
    check_bpp = 0.10 <= test_bpp_avg <= 0.15
    checks.append(("BPP in range (0.10-0.15)", test_bpp_avg, "0.10-0.15", check_bpp, True))
    
    # Print check results
    required_passed = 0
    required_total = 0
    optional_passed = 0
    optional_total = 0
    
    for check_name, current_val, target_val, passed, is_required in checks:
        status = "PASS" if passed else "FAIL"
        req_mark = "[REQ]" if is_required else "[OPT]"
        
        if isinstance(current_val, float):
            if isinstance(target_val, str):
                print(f"  {status:4s} {req_mark} {check_name}: {current_val:.4f}")
            else:
                print(f"  {status:4s} {req_mark} {check_name}: {current_val:.2f} / {target_val}")
        else:
            print(f"  {status:4s} {req_mark} {check_name}")
        
        if is_required:
            required_total += 1
            if passed:
                required_passed += 1
        else:
            optional_total += 1
            if passed:
                optional_passed += 1
    
    print()
    print(f"{'='*70}")
    print(f"Required Checks Passed: {required_passed}/{required_total}")
    print(f"Optional Checks Passed: {optional_passed}/{optional_total}")
    print(f"{'='*70}\n")
    
    # Determine if can transition
    can_transition_min = required_passed == required_total
    can_transition_recommended = can_transition_min and (optional_passed >= optional_total * 0.6)
    
    if can_transition_recommended:
        print("SUCCESS! Stage 1 training complete, ready for Stage 2!")
        print()
        print("Next Steps:")
        print("  1. Verify best model exists:")
        print("     pretrained/DMC_WAN/1/checkpoint_best_stage1_wan.pth.tar")
        print()
        print("  2. Start Stage 2 training with command shown in README")
        print()
    elif can_transition_min:
        print("WARNING: Stage 1 meets minimum requirements but not all targets")
        print()
        print(f"  Current Status:")
        print(f"    - Test PSNR: {test_psnr_avg:.2f} dB (Target: 26 dB)")
        print(f"    - Train PSNR: {train_psnr_avg:.2f} dB (Target: 27 dB)")
        print()
        print(f"  Recommendation:")
        print(f"    - Continue training for 10-15 more epochs")
        print(f"    - PSNR needs to improve by {26.0 - test_psnr_avg:.1f} dB")
        print()
    else:
        print("INCOMPLETE: Stage 1 not ready for transition, continue training")
        print()
        print(f"  Gap Analysis:")
        if test_psnr_avg < 25.0:
            print(f"    - Test PSNR: {test_psnr_avg:.2f} dB, need {25.0 - test_psnr_avg:.1f} dB more")
        if train_psnr_avg < 26.0:
            print(f"    - Train PSNR: {train_psnr_avg:.2f} dB, need {26.0 - train_psnr_avg:.1f} dB more")
        if train_latent_mse_avg >= 0.15:
            print(f"    - LatentMSE: {train_latent_mse_avg:.4f}, need < 0.15")
        
        # Estimate remaining epochs
        if test_psnr_avg < 25.0 and len(test_results) >= 10:
            recent_10 = test_results[-10:]
            psnr_improvement = recent_10[-1]['psnr'] - recent_10[0]['psnr']
            epochs_per_db = 10 / max(psnr_improvement, 0.1)
            remaining_db = 25.0 - test_psnr_avg
            estimated_epochs = int(remaining_db * epochs_per_db)
            print()
            print(f"  Estimated remaining epochs: {estimated_epochs}")
            print(f"    (Based on recent 10-epoch improvement: {psnr_improvement:.2f} dB)")
        print()
    
    print(f"{'='*70}\n")
    
    return can_transition_recommended

def main():
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        # Find latest log file
        log_dir = "pretrained/DMC_WAN/1"
        if os.path.exists(log_dir):
            log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
            if log_files:
                log_files.sort(reverse=True)
                log_file = os.path.join(log_dir, log_files[0])
            else:
                print("Error: No log files found")
                sys.exit(1)
        else:
            print(f"Error: Log directory not found: {log_dir}")
            sys.exit(1)
    
    print(f"\nAnalyzing log file: {log_file}")
    
    train_results, test_results = parse_log_file(log_file)
    
    if not test_results:
        print("Error: No test results found")
        sys.exit(1)
    
    print(f"Successfully parsed {len(train_results)} training epochs, {len(test_results)} test epochs")
    
    # Check Stage 1 transition conditions
    check_stage1_transition(train_results, test_results)

if __name__ == "__main__":
    main()
