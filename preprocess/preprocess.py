import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess(dataset):
    path = '../raw/' + dataset
    activity = pd.read_csv(path + '_activity.csv')
    new_activity = activity.drop(['char_id', 'server'], axis=1)
    new_activity = new_activity.sort_values(['acc_id', 'day'])
    new_activity = new_activity.reset_index(drop=True)
    new_activity = new_activity.groupby(['acc_id', 'day']).sum()
    new_activity = pd.DataFrame(new_activity)


    trade = pd.read_csv(path + '_trade.csv')
    new_trade = trade.drop(['time', 'type', 'item_type', 'item_amount', 'item_price', 'server', 'source_char_id', 'target_char_id'], axis=1)
    new_trade_sell = new_trade.sort_values(['source_acc_id', 'day'])
    new_trade_sell.columns = ['day', 'acc_id', 'num_sell']
    new_trade_sell = new_trade_sell.reset_index(drop=True)
    new_trade_sell = new_trade_sell.groupby(['acc_id', 'day']).count()
    new_trade_sell = pd.DataFrame(new_trade_sell)


    trade = pd.read_csv(path + '_trade.csv')
    new_trade = trade.drop(['time', 'type', 'item_type', 'item_amount', 'item_price', 'server', 'source_char_id', 'target_char_id'], axis=1)
    new_trade_buy = new_trade.sort_values(['target_acc_id', 'day'])
    new_trade_buy.columns = ['day', 'num_buy', 'acc_id']
    new_trade_buy = new_trade_buy.reset_index(drop=True)
    new_trade_buy = new_trade_buy.groupby(['acc_id', 'day']).count()
    new_trade_buy = pd.DataFrame(new_trade_buy)


    new_trade = pd.merge(new_trade_sell, new_trade_buy, left_index=True, right_index=True, how='outer')


    combat = pd.read_csv(path + '_combat.csv')
    new_combat = combat.drop(['char_id', 'server', 'class'], axis=1)
    new_combat = new_combat.sort_values(['acc_id', 'day'])
    new_combat = new_combat.reset_index(drop=True)
    new_combat = new_combat.groupby(['acc_id', 'day']).agg({'level': 'max', 'pledge_cnt':'sum', 'random_attacker_cnt': 'sum', 'random_defender_cnt':'sum', 'temp_cnt':'sum', 'same_pledge_cnt':'sum', \
                                                           'etc_cnt':'sum', 'num_opponent':'sum'})
    new_combat = pd.DataFrame(new_combat)


    pledge = pd.read_csv(path + '_pledge.csv')
    new_pledge = pledge.drop(['char_id', 'server'], axis=1)
    new_pledge = new_pledge.sort_values(['acc_id', 'day'])
    new_pledge = new_pledge.reset_index(drop=True)
    new_pledge = new_pledge.groupby(['acc_id', 'day']).agg({'pledge_id':lambda x:x.value_counts().index[0], 'play_char_cnt':'mean', 'combat_char_cnt': 'mean', 'pledge_combat_cnt':'sum', 'random_attacker_cnt':'sum', 'random_defender_cnt':'sum', \
                                                           'same_pledge_cnt':'sum', 'temp_cnt':'sum', 'etc_cnt':'sum', 'combat_play_time':'sum', 'non_combat_play_time':'sum'})
    new_pledge = pd.DataFrame(new_pledge)


    payment = pd.read_csv(path + '_payment.csv')
    new_payment = payment.sort_values(['acc_id', 'day'])
    new_payment = new_payment.reset_index(drop=True)
    new_payment = new_payment.groupby(['acc_id', 'day']).sum()
    new_payment = pd.DataFrame(new_payment)


    new_data = pd.merge(new_activity, new_trade, left_index=True, right_index=True, how='left')
    new_data = pd.merge(new_data, new_combat, left_index=True, right_index=True, how='outer')
    new_data = pd.merge(new_data, new_pledge, left_index=True, right_index=True, how='outer')
    new_data = pd.merge(new_data, new_payment, left_index=True, right_index=True, how='outer')
    new_data = new_data.fillna(0)
    new_data = new_data.reset_index()

    # new_data


    new_data = new_data.groupby(['acc_id']).agg({new_data.columns[1]:lambda x: sum(set(x)), 'playtime': 'sum', 'npc_kill':'sum', 'solo_exp':'sum', 'party_exp':'sum',
           'quest_exp':'sum', 'rich_monster':'sum', 'death':'sum', 'revive':'sum', 'exp_recovery':'sum',
           'fishing':'sum', 'private_shop':'sum', 'game_money_change':'sum', 'enchant_count':'sum',
           'num_sell':'sum', 'num_buy':'sum', 'level':'max', 'pledge_cnt':'sum', 'random_attacker_cnt_x':'sum',
           'random_defender_cnt_x':'sum', 'temp_cnt_x':'sum', 'same_pledge_cnt_x':'sum', 'etc_cnt_x':'sum',
           'num_opponent':'sum', 'pledge_id':'sum', 'play_char_cnt':'mean', 'combat_char_cnt':'mean',
           'pledge_combat_cnt':'sum', 'random_attacker_cnt_y':'sum', 'random_defender_cnt_y':'sum',
           'same_pledge_cnt_y':'sum', 'temp_cnt_y':'sum', 'etc_cnt_y':'sum', 'combat_play_time':'sum',
           'non_combat_play_time':'sum', 'amount_spent':'sum'})

    scaled_train = new_data.copy()
    col_names = ['day', 'playtime', 'npc_kill', 'solo_exp', 'party_exp', 'quest_exp', 'rich_monster',
            'death', 'revive', 'exp_recovery', 'fishing',
           'private_shop', 'game_money_change', 'enchant_count', 'num_sell',
           'num_buy',  'pledge_cnt', 'random_attacker_cnt_x',
           'random_defender_cnt_x', 'temp_cnt_x', 'same_pledge_cnt_x', 'etc_cnt_x',
           'num_opponent', 'pledge_id', 'play_char_cnt', 'combat_char_cnt',
           'pledge_combat_cnt', 'random_attacker_cnt_y', 'random_defender_cnt_y',
           'same_pledge_cnt_y', 'temp_cnt_y', 'etc_cnt_y', 'combat_play_time',
           'non_combat_play_time', 'amount_spent']
           
    features = scaled_train[col_names]
    scaler = MinMaxScaler((-1,1)).fit(features.values)
    features = scaler.transform(features.values)
    scaled_train[col_names] = features
    scaled_train = pd.DataFrame(scaled_train, columns = new_data.columns)
    scaled_train[['level']] = scaled_train[['level']].astype(int) # int

    scaled_train.to_csv(dataset + "_preprocess_1.csv", mode='w', header=False)

    if dataset == "train":
        label = pd.read_csv(path + '_label.csv')
        new_label = label.sort_values(['acc_id'])
        new_label = pd.DataFrame(new_label)
        new_label = new_label.reset_index(drop=True)

        new_label.to_csv( dataset + "_label_1.csv", mode='w', index=False, header=False)
    
if __name__ == "__main__":
    print("The training data is being preprocessed.")
    preprocess("train")
    print("The test1 data is being preprocessed.")
    preprocess("test1")
    print("The test2 data is being preprocessed.")
    preprocess("test2")
    print("Finish")