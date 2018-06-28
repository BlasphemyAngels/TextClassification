from utils import str_to_list

def textcnn(args):

    config = args["config"]

    config["filters_size"] = str_to_list(config["filters_size"])
    config["filters_num"] = str_to_list(config["filters_num"])
    config["l2_reg_lambda"] = float(config["l2_reg_lambda"])
    config["dropout_prob"] = float(config["dropout_prob"])
    config["class_nums"] = int(config["class_nums"])
    config["init_learning_rate"] = float(config["init_learning_rate"])
    config["learning_rate_decay"] = float(config["learning_rate_decay"])

    return config

def main(args):
    print(args)
    config = args["config"]

    print(config)
    config["text_length"] = int(config["text_length"])
    config["vocab_size"] = int(config["vocab_size"])
    config["embedding_size"] = int(config["vocab_size"])
    config["batch_size"] = int(config["batch_size"])
    config["epoch"] = int(config["epoch"])
    config["label_smooth_eps"] = float(config["label_smooth_eps"])
    return config
