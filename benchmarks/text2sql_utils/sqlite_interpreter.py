import sys
import sqlite3
from func_timeout import func_timeout, FunctionTimedOut

def execute_sql(predicted_sql, ground_truth, db_path, show_num_rows=10):
    conn = sqlite3.connect(db_path)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    pred_md_table = "| " + " | ".join(["Column{}".format(i + 1) for i in range(len(predicted_res[0]))]) + " |\n"
    pred_md_table += "| " + " | ".join(["---" for _ in range(len(predicted_res[0]))]) + " |\n"
    for row in predicted_res[:show_num_rows]:
        pred_md_table += "| " + " | ".join(map(str, row)) + " |\n"
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()

    gt_md_table = "| " + " | ".join(["Column{}".format(i + 1) for i in range(len(ground_truth_res[0]))]) + " |\n"
    gt_md_table += "| " + " | ".join(["---" for _ in range(len(ground_truth_res[0]))]) + " |\n"
    for row in ground_truth_res[:show_num_rows]:
        gt_md_table += "| " + " | ".join(map(str, row)) + " |\n"
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    
    return {
        "res": res, 
        "predicted_res": predicted_res,
        "ground_truth_res": ground_truth_res,
        "pred_md_table": pred_md_table, 
        "gt_md_table": gt_md_table
    }

def execute_model(predicted_sql, ground_truth, db_path, meta_time_out=30):
    # Initialize results
    result = {
        "res": 0,
        "predicted_res": '',
        "ground_truth_res": '',
        "pred_md_table": '',
        "gt_md_table": ''
    }

    try:
        # Execute SQL with timeout
        res_dict = func_timeout(
            meta_time_out, execute_sql,
            args=(predicted_sql, ground_truth, db_path)
        )
        
        # Update result dictionary with returned values
        result.update({
            "res": res_dict.get('res', 0),
            "predicted_res": res_dict.get('predicted_res', ''),
            "ground_truth_res": res_dict.get('ground_truth_res', ''),
            "pred_md_table": res_dict.get('pred_md_table', ''),
            "gt_md_table": res_dict.get('gt_md_table', '')
        })
        
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result["predicted_res"] = 'Database execution timeout'
    except Exception as e:
        result["predicted_res"] = str(e)
    
    return result
