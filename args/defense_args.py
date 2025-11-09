import os 


def get_defense_args(p, defense):
    p.add_argument(
        "--df_none", dest="df.none", default="none", type=str, help="No defense selected.")
    
    if defense == "Summary":
        p.add_argument(
            "--df_summary_prompt", dest="df.summary_prompt", default=None, type=str, help="Path to the summary prompt template.")
    if defense == "Threshold":
        p.add_argument(
            "--df_threshold", dest="df.threshold", default=0.5, type=float, help="Threshold for filtering responses.")
    if defense == "SystemBlock":
        p.add_argument(
            "--df_system_block", dest="df.system_block", default=None, type=str, help="Path to the system block template.")
    return p
    
    