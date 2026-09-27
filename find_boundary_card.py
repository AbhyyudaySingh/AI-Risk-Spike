import pandas as pd

X_train = pd.read_parquet("data/processed/features/X_train.parquet")
X_val = pd.read_parquet("data/processed/features/X_val.parquet")

# We want a card1 that exists in both train and val
train_cards = set(X_train['card1'].unique())
val_cards = set(X_val['card1'].unique())
overlap = train_cards.intersection(val_cards)

# Pick one overlapping card
for card in overlap:
    if card != 'Unknown':
        # Get the last transaction for this card in train
        t_rows = X_train[X_train['card1'] == card]
        v_rows = X_val[X_val['card1'] == card]
        
        # We want a card that had some decent history in train
        if len(t_rows) > 5 and len(v_rows) > 0:
            last_train = t_rows.iloc[-1]
            first_val = v_rows.iloc[0]
            
            print(f"--- Example for card1: {card} ---")
            print("LAST TRAIN RECORD:")
            print(f"TransactionID: {last_train['TransactionID']}")
            print(f"TransactionDT: {last_train['TransactionDT']}")
            print(f"Prior Tx Count: {last_train['history_card1_prior_tx_count']}")
            print(f"Current Tx Amt: {last_train['TransactionAmt']}")
            
            print("\nFIRST VAL RECORD:")
            print(f"TransactionID: {first_val['TransactionID']}")
            print(f"TransactionDT: {first_val['TransactionDT']}")
            print(f"Prior Tx Count: {first_val['history_card1_prior_tx_count']}")
            break
