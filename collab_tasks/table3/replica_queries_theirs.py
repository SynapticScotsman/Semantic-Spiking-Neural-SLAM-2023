# Descriptive queries are sampled from clip_v_llm_exps/replica_mturk_annotated_scenegraph_nodes/*.json

REPLICA_DESC_QUERIES = None


# https://github.com/alik-git/CFSLAM/blob/ali_debug/cfslam%2Fclip_v_llm_exps%2Frun_queries_affordances_replica.py#L148

REPLICA_OFFICE_AFFORD_QUERIES = {
    "1": {
        "query_text": "Something to watch the news on"
    },
    "2": {
        "query_text": "Something to tell the time"
    },
    "3": {
        "query_text": "Something comfortable to sit on"
    },
    "4": {
        "query_text": "Something to dispose of wastepaper in"
    },
    "5": {
        "query_text": "Something to add light into the room"
    },
}

REPLICA_ROOM_AFFORD_QUERIES = {
    "1": {
        "query_text": "Somewhere to store decorative cups"
    },
    "2": {
        "query_text": "Something to add light into the room"
    },
    "3": {
        "query_text": "Somewhere to set food for dinner"
    },
    "4": {
        "query_text": "Something I can open with my keys"
    },
    "5": {
        "query_text": "Something to sit upright for a work call"
    },
}


# https://github.com/alik-git/CFSLAM/blob/ali_debug/cfslam/clip_v_llm_exps/run_queries_negation_replica.py#L148

REPLICA_OFFICE_NEGATE_QUERIES = {
    "1": {
        "query_text": "Something to sit on other than a chair"
    },
    "2": {
        "query_text": "Something very heavy, unlike a clock"
    },
    "3": {
        "query_text": "Something rigid, unlike a cushion"
    },
    "4": {
        "query_text": "Something small, unlike a couch"
    },
    "5": {
        "query_text": "Something light, unlike a table"
    },
}

REPLICA_ROOM_NEGATE_QUERIES = {
    "1": {
        "query_text": "Something small, unlike a cabinet"
    },
    "2": {
        "query_text": "Something light, unlike a table"
    },
    "3": {
        "query_text": "Something soft, unlike a table"
    },
    "4": {
        "query_text": "Something not transparent, unlike a window"
    },
    "5": {
        "query_text": "Something rigid, unlike a rug"
    },
}

