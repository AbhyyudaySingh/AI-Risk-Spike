import pandas as pd

X_val = pd.read_parquet("data/processed/features/X_val.parquet")
X_test = pd.read_parquet("data/processed/features/X_test.parquet")

val_cards = set(X_val['card1'].unique())
test_cards = set(X_test['card1'].unique())
overlap = val_cards.intersection(test_cards)

for card in overlap:
    if card != 'Unknown':
        v_rows = X_val[X_val['card1'] == card]
        t_rows = X_test[X_test['card1'] == card]
        
        if len(v_rows) > 5 and len(t_rows) > 0:
            last_val = v_rows.iloc[-1]
            first_test = t_rows.iloc[0]
            
            print(f"--- Example for card1: {card} ---")
            print("LAST VALIDATION RECORD:")
            print(f"TransactionID: {last_val['TransactionID']}")
            print(f"TransactionDT: {last_val['TransactionDT']}")
            print(f"Prior Tx Count: {last_val['history_card1_prior_tx_count']}")
            print(f"Current Tx Amt: {last_val['TransactionAmt']}")
            
            print("\nFIRST TEST RECORD:")
            print(f"TransactionID: {first_test['TransactionID']}")
            print(f"TransactionDT: {first_test['TransactionDT']}")
            print(f"Prior Tx Count: {first_test['history_card1_prior_tx_count']}")
            break
