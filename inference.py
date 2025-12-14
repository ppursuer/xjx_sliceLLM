import torch
import time

def forward_full(model, input_ids, tokenizer, collector_full):
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
    hidden_states = model.model.embed_tokens(input_ids)

    start_time = time.time()
    for i, layer in enumerate(model.model.layers):
        h_in = hidden_states[:, -1, :].detach()

        hidden_states = layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            use_cache=False
        )[0]
        h_out = hidden_states[:, -1, :].detach()

        # 当前层 top1 token
        logits = model.lm_head(model.model.norm(h_out))
        token_str = tokenizer.decode([torch.argmax(logits, -1).item()])
        l2 ,cos = collector_full.compute_metric(h_in, h_out)
        collector_full.add_records("full", i, token_str, l2, cos)

    end_time = time.time()
    total_time = end_time - start_time
    final_logits = model.lm_head(model.model.norm(hidden_states[:, -1, :]))

    probs = torch.softmax(final_logits, dim=-1)
    max_prob, next_token_id = torch.max(probs, dim=-1)

    # confidence = max_prob.item()  
    input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

    return next_token_id, token_str, input_ids, total_time

def forward_skip(model, input_ids, tokenizer, collector_skip, start_layer, end_layer, min_start_layer):
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
    hidden_states = model.model.embed_tokens(input_ids)

    return_layer = start_layer
    once = False
    start_time = time.time()
    ################ 浅层 #################
    for i in range(start_layer):
        if i == return_layer:
            break

        layer = model.model.layers[i]
        h_in = hidden_states[:, -1, :].detach()
        hidden_states = layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            use_cache=False,
        )[0]
        h_out = hidden_states[:, -1, :].detach()

        logits = model.lm_head(model.model.norm(h_out))
        token_str = tokenizer.decode([torch.argmax(logits, -1).item()])
        LRS = collector_skip.compute("skip", i, h_in, h_out, token_str)

        # 决策跳跃 
        # if LRS <= 0.2 and once is False: 
        #     once = True 
        #     return_layer = start_layer - 1 
        #     start_layer = return_layer

        if (LRS <= 0.2) and (not once):

            once = True
            # -------------------------
            # 降低 start_layer
            # -------------------------
            dynamic_start = max(start_layer - 1, min_start_layer)

            # 更新跳跃区间
            start_layer = dynamic_start

            return_layer = dynamic_start  # for log/debug

    ################ skip layer #################  
    # for i in range(start_layer, end_layer):
    #     collector_skip.compute("skip", i, h_in, h_out, token_str)    

    #################  deep layer ################

    for i in range(end_layer, len(model.model.layers)):
        layer = model.model.layers[i]
        h_in = hidden_states[:, -1, :].detach()

        hidden_states = layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            use_cache=False,
        )[0]

        h_out = hidden_states[:, -1, :].detach()
        logits = model.lm_head(model.model.norm(h_out))
        token_str = tokenizer.decode([torch.argmax(logits, -1).item()])
        collector_skip.compute("skip", i, h_in, h_out, token_str)
    
    end_time = time.time()
    total_time = end_time - start_time
    final_logits = model.lm_head(model.model.norm(hidden_states[:, -1, :]))
    probs = torch.softmax(final_logits, dim=-1)
    max_prob, next_token_id = torch.max(probs, dim=-1)

    # confidence = max_prob.item()  
    input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
    return next_token_id, token_str, input_ids, return_layer, end_layer - start_layer, total_time

def forward_skip_update(model, input_ids, tokenizer, collector_skip, start_layer, end_layer, min_start_layer, LRS_threshold):
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
    hidden_states = model.model.embed_tokens(input_ids)

    return_layer = start_layer
    buffer_cnt = 0  # 连续满足 LRS <= 0.2 的层数计数

    start_time = time.time()
    ################ 浅层 #################
    for i in range(start_layer):

        layer = model.model.layers[i]
        h_in = hidden_states[:, -1, :].detach()

        hidden_states = layer(hidden_states,
                              attention_mask=None,
                              position_ids=position_ids,
                              use_cache=False)[0]

        h_out = hidden_states[:, -1, :].detach()
        logits = model.lm_head(model.model.norm(h_out))
        token_str = tokenizer.decode([torch.argmax(logits, -1).item()])

        LRS = collector_skip.compute("skip", i, h_in, h_out, token_str)
        # print(f"layer:{i+1} LRS:", LRS)
        # ===========================================================
        #   判定：只有 BUFFER 区域所有层都满足 LRS < 0.2 才能下降
        # ===========================================================

        # 只有在缓冲区（start_layer-1 到 min_start_layer）才检查
        if min_start_layer <= i < start_layer:
            # print(f"layer:{i+1} LRS:", LRS)
            if LRS <= 0.175:
                buffer_cnt += 1
            else:
                buffer_cnt = 0

            # 若所有 buffer 层都满足条件，则下降 start_layer
            BUFFER_SIZE = start_layer - min_start_layer
            if buffer_cnt >= BUFFER_SIZE:
                # print("满足条件")
                start_layer -= 1      # 放慢下降节奏（只下降一层）
                return_layer = start_layer
                buffer_cnt = 0        # 重置
        # ===========================================================

    #################  deep layers ################
    for i in range(end_layer, len(model.model.layers)):
        layer = model.model.layers[i]
        h_in = hidden_states[:, -1, :].detach()
        hidden_states = layer(hidden_states,
                              attention_mask=None,
                              position_ids=position_ids,
                              use_cache=False)[0]
        h_out = hidden_states[:, -1, :].detach()
        logits = model.lm_head(model.model.norm(h_out))
        token_str = tokenizer.decode([torch.argmax(logits, -1).item()])
        collector_skip.compute("skip", i, h_in, h_out, token_str)

    end_time = time.time()
    total_time = end_time - start_time
    final_logits = model.lm_head(model.model.norm(hidden_states[:, -1, :]))
    probs = torch.softmax(final_logits, dim=-1)
    max_prob, next_token_id = torch.max(probs, dim=-1)

    input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

    return next_token_id, token_str, input_ids, return_layer, end_layer - start_layer, total_time
