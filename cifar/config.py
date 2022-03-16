# -*- coding: utf-8 -*-
import argparse


# +
def parse_opt():
    parser = argparse.ArgumentParser()

    # Order
    parser.add_argument('--ORDER', type=str, default='random')
    # Percentage
    parser.add_argument('--PERCENTAGE', type=int, default=0)
    # Seed
    parser.add_argument('--SEED', type=int, default=42)
    
    # Data Arguments
    parser.add_argument('--DATA_PATH', type=str, default='data')

#     parser.add_argument('--TRAIN_DATA', type=str, default='train.csv')
    parser.add_argument('--TRAIN_DEV_DATA', type=str, default='train_10000.csv')
    parser.add_argument('--DEV_DATA', type=str, default='dev_5000.csv')
    parser.add_argument('--TEST_DATA', type=str, default='test.csv')
    
    
    # Model Arguments    
    parser.add_argument('--HIDDEN_DIM', type=int, default=2048)
    parser.add_argument('--NUM_LABELS', type=int, default=10)
    
    # Training Arguments
    parser.add_argument('--EPOCH', type=int, default=10)

    parser.add_argument('--TRAIN_BATCH_SIZE', type=int, default=32)
    parser.add_argument('--TEST_BATCH_SIZE', type=int, default=32)
    
    parser.add_argument('--LEARNING_RATE', type=float, default=1e-2)
    parser.add_argument('--MOMENTUM', type=float, default=0.9)
    
    parser.add_argument('--L2_LAMBDA', type=float, default=5e-3)
    
    # Save Path
    parser.add_argument('--OUTPUT', type=str, default="saved")
    parser.add_argument('--SAVE_CHECKPOINT', type=bool, default=True)
    
    args = parser.parse_args() 
    return args


# +
def parse_opt_if_attr():
    parser = argparse.ArgumentParser()

    # Order
    parser.add_argument('--ORDER', type=str, default='random')
    # Percentage
    parser.add_argument('--PERCENTAGE', type=int, default=0)
    # Seed
    parser.add_argument('--SEED', type=int, default=42)
    parser.add_argument('--CHECKPOINT', type=int, default=42)
    #
    # Data Arguments
    parser.add_argument('--DATA_PATH', type=str, default='data')

#     parser.add_argument('--TRAIN_DATA', type=str, default='train.csv')
    parser.add_argument('--TRAIN_DEV_DATA', type=str, default='train_10000.csv')
    parser.add_argument('--DEV_DATA', type=str, default='dev_5000.csv')
    parser.add_argument('--TEST_DATA', type=str, default='test.csv')
        
    # Model Arguments

    parser.add_argument('--HIDDEN_DIM', type=int, default=2048)
    parser.add_argument('--NUM_LABELS', type=int, default=10)
    # Training Arguments
    parser.add_argument('--EPOCH', type=int, default=10)

    parser.add_argument('--TRAIN_BATCH_SIZE', type=int, default=32)
    parser.add_argument('--TEST_BATCH_SIZE', type=int, default=32)
    
    parser.add_argument('--LEARNING_RATE', type=float, default=1e-2)
    parser.add_argument('--MOMENTUM', type=float, default=0.9)
    
    parser.add_argument('--L2_LAMBDA', type=float, default=5e-3)
    
    # Save Path
    parser.add_argument('--OUTPUT', type=str, default="saved")
    parser.add_argument('--SAVE_CHECKPOINT', type=bool, default=True)
    
    # IF Arguments
    parser.add_argument('--DAMP', type=float, default=5e-3)
    parser.add_argument('--SCALE', type=int, default=1e4)
    parser.add_argument('--NUM_SAMPLES', type=int, default=1000)
    
    # Others
    parser.add_argument('--START', type=int, default=0)
    parser.add_argument('--LENGTH', type=int, default=1000)
    
    args = parser.parse_args() 
    
    return args

# -



# +
def parse_opt_if():
    parser = argparse.ArgumentParser()

    # Order
    parser.add_argument('--ORDER', type=str, default='random')
    # Percentage
    parser.add_argument('--PERCENTAGE', type=int, default=0)
    # Seed
    parser.add_argument('--SEED', type=int, default=42)
    parser.add_argument('--CHECKPOINT', type=int, default=42)
    #
    # Data Arguments
    parser.add_argument('--DATA_PATH', type=str, default='data')

#     parser.add_argument('--TRAIN_DATA', type=str, default='train.csv')
    parser.add_argument('--TRAIN_DEV_DATA', type=str, default='attr.csv')
    parser.add_argument('--DEV_DATA', type=str, default='dev_5000.csv')
    parser.add_argument('--TEST_DATA', type=str, default='test.csv')
        
    # Model Arguments

    parser.add_argument('--HIDDEN_DIM', type=int, default=2048)
    parser.add_argument('--NUM_LABELS', type=int, default=10)
    # Training Arguments
    parser.add_argument('--EPOCH', type=int, default=10)

    parser.add_argument('--TRAIN_BATCH_SIZE', type=int, default=32)
    parser.add_argument('--TEST_BATCH_SIZE', type=int, default=32)
    
    parser.add_argument('--LEARNING_RATE', type=float, default=1e-2)
    parser.add_argument('--MOMENTUM', type=float, default=0.9)
    
    parser.add_argument('--L2_LAMBDA', type=float, default=5e-3)
    
    # Save Path
    parser.add_argument('--OUTPUT', type=str, default="saved")
    parser.add_argument('--SAVE_CHECKPOINT', type=bool, default=True)
    
    # IF Arguments
    parser.add_argument('--DAMP', type=float, default=5e-3)
    parser.add_argument('--SCALE', type=int, default=1e4)
    parser.add_argument('--NUM_SAMPLES', type=int, default=1000)
    
    # Attr Arguments
    parser.add_argument('--ATTR_ORDER', type=str, default='random')
    parser.add_argument('--ATTR_PERCENTAGE', type=int, default='0')
    
    # Others
    parser.add_argument('--START', type=int, default=0)
    parser.add_argument('--LENGTH', type=int, default=1000)
    
    args = parser.parse_args() 
    
    return args

# -




