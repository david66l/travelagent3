"""Reconcile recorded steps with explicitly retained, uncommitted proposals."""
CONTRACT='episode-accounting.v1'


def verify_episode_accounting(episode):
    errors=[];uncommitted_steps=0;action_ids=set()
    if sum(e.event_type=='agent_uncommitted_batch' for e in episode.events)>1:
        errors.append('MULTIPLE_UNCOMMITTED_TERMINAL_BATCHES')
    for event in episode.events:
        if event.event_type!='agent_uncommitted_batch':continue
        payload=event.payload
        if payload.get('schema_version')!='uncommitted-batch.v1':
            errors.append('UNCOMMITTED_BATCH_SCHEMA_INVALID');continue
        if episode.status!='failed' or payload.get('training_labels_authorized') is not False:
            errors.append('UNCOMMITTED_BATCH_TERMINAL_INVALID')
        before=payload.get('budget_before') or {};after=payload.get('budget_after') or {}
        if episode.final_state is None or after!=episode.final_state.get('budget'):
            errors.append('UNCOMMITTED_FINAL_BUDGET_MISMATCH')
        attempts=payload.get('policy_attempts') or []
        returned=[v for v in attempts if v.get('proposal')]
        for item in returned:
            aid=item['proposal'].get('action_id')
            if not aid or aid in action_ids:errors.append('UNCOMMITTED_ACTION_ID_INVALID')
            action_ids.add(aid)
        try:
            charged=after['used_episode_steps']-before['used_episode_steps']
            if not 0<=charged<=len(returned):errors.append('UNCOMMITTED_STEP_CHARGE_INVALID')
            else:uncommitted_steps+=charged
            if any(after[k]<before[k] for k in ['used_episode_steps','used_tool_calls','used_solver_calls','used_tokens','used_latency_ms']):
                errors.append('UNCOMMITTED_BUDGET_DECREASED')
        except (KeyError,TypeError):errors.append('UNCOMMITTED_BUDGET_MISSING')
        if any(v.get('status') in {'pending','running'} for v in payload.get('execution_attempts',[])):
            errors.append('UNCOMMITTED_EXECUTOR_NOT_DRAINED')
        if any(v.get('status') in {'pending','running'} for v in attempts):
            errors.append('UNCOMMITTED_POLICY_NOT_DRAINED')
    if any(s.action.action_id in action_ids for s in episode.steps):errors.append('UNCOMMITTED_ACTION_ALSO_RECORDED')
    if episode.final_state is not None:
        initial=episode.initial_state['budget']['used_episode_steps'];final=episode.final_state['budget']['used_episode_steps']
        if final-initial!=len(episode.steps)+uncommitted_steps:errors.append('EPISODE_STEP_ACCOUNTING_MISMATCH')
    return errors
