class Preprocessor:
    def reset(self):
        self.buff_cnt=0
        self.buff_his = []

    def get_buff_cnt(self,hero,extra_info):
        # buff获取次数
        self.buff_his.append(hero["speed_up"])
        if extra_info:
            self.buff_cnt = extra_info["game_info"]["buff_count"]
        else:
            if(len(self.buff_his)>1):
                if(self.buff_his[-1]==1 and self.buff_his[-2]==0):
                    self.buff_cnt+=1

    def pb2struct(self, frame_state, last_action):
        obs, extra_info = frame_state
        hero = obs["frame_state"]["heroes"][0]
        # buff获取次数
        self.get_buff_cnt(hero, extra_info)

    def process(self, frame_state, last_action):
        # buff次数
        buff_cnt = self.buff_cnt
        return (
            feature,
            legal_action,
            reward_process(
                self.feature_end_pos[-1],
                self.feature_history_pos[-1],
                buff_cnt,
            ),

def reward_process(end_dist, history_dist,buff_cnt):
    # buff奖励
    buff_reward = 0.05*math.pow(0.5, buff_cnt)
    return [step_reward + dist_reward + end_reward + buff_reward ]