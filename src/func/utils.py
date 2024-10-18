import wandb


# 플롯 제목 통일을 위해 LGBM 커스텀 콜백 설정
def lgb_wandb_callback():
    def callback(env):
        for data_name, metric_name, value, _ in env.evaluation_result_list:
            # 메트릭 이름 변경
            if metric_name == "l1":
                metric_name = "MAE"
            elif metric_name == "rmse":
                metric_name = "RMSE"

            # 로그 이름 생성
            log_name = f"{data_name}-{metric_name}"

            # wandb에 로그 기록
            wandb.log({log_name: value}, step=env.iteration)

    return callback
