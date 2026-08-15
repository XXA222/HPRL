(() => {
  "use strict";

  let auth = "";
  let timer = null;
  let allJobs = [];
  let allPipelines = [];
  let selectedJobId = "";
  let selectedPipelineId = "";
  const selectedMetric = {};
  const $ = (id) => document.getElementById(id);
  const escapeHtml = (value) => String(value ?? "").replace(
    /[&<>"']/g,
    (char) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[char],
  );

  async function api(path, options = {}) {
    const response = await fetch("/api/v1" + path, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(auth ? {Authorization: auth} : {}),
        ...(options.headers || {}),
      },
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    return response.json();
  }

  const percent = (value) => Math.round(Number(value || 0) * 100);
  const formatBytes = (value) => {
    let size = Number(value || 0);
    const units = ["B", "KB", "MB", "GB", "TB"];
    let index = 0;
    while (size >= 1024 && index < units.length - 1) {
      size /= 1024;
      index += 1;
    }
    return `${size.toFixed(index > 1 ? 1 : 0)} ${units[index]}`;
  };

  function renderCards(data) {
    const states = data.states || {};
    const executor = data.executor || {};
    const system = executor.system || {};
    const gpu = (system.gpus || [])[0] || {};
    const rows = [
      ["任务总数", data.total || 0],
      ["运行中", states.RUNNING || 0],
      ["已暂停", states.PAUSED || 0],
      ["排队", executor.queued_count || 0],
      ["已完成", states.SUCCEEDED || 0],
      ["失败", states.FAILED || 0],
      ["CPU", `${Number(system.cpu_percent || 0).toFixed(0)}%`],
      ["内存", `${Number(system.memory_percent || 0).toFixed(0)}%`],
      ["GPU", gpu.utilization_percent == null ? "N/A" : `${gpu.utilization_percent}%`],
    ];
    $("cards").innerHTML = rows.map(
      ([name, value]) => `<div class="card"><span>${name}</span><b>${value}</b></div>`,
    ).join("");
  }

  function renderResources(executor) {
    const system = executor.system || {};
    const gpus = system.gpus || [];
    const rows = [
      ["CPU", `${Number(system.cpu_percent || 0).toFixed(1)}%`],
      ["RAM", `${formatBytes(system.memory_used_bytes)} / ${formatBytes(system.memory_total_bytes)}`],
      ["队列", executor.queued_count || 0],
      ["活动", `${executor.active_count || 0}/${executor.max_concurrent || 0}`],
      ["CPU训练槽", `${executor.max_cpu_training_jobs ?? 0} · ${executor.cpu_threads_per_job ?? 0} threads/job`],
      ["CPU线程预算", `${executor.effective_cpu_thread_limit ?? 0}`],
      ["GPU训练槽", `${executor.max_gpu_jobs ?? 0}`],
    ];
    gpus.forEach((gpu) => rows.push([
      `GPU ${gpu.index}`,
      `${gpu.utilization_percent}% · ${gpu.memory_used_mb}/${gpu.memory_total_mb} MB · ${gpu.temperature_c}°C`,
    ]));
    $("resourceCards").innerHTML = rows.map(
      ([name, value]) => `<div class="resource"><span>${escapeHtml(name)}</span><b>${escapeHtml(value)}</b></div>`,
    ).join("");
    $("executorSummary").textContent = executor.paused
      ? "队列已暂停"
      : `活动 ${executor.active_count || 0} · 排队 ${executor.queued_count || 0}`;
    $("executor").textContent = JSON.stringify(executor, null, 2);
  }

  function renderJobs() {
    const kind = $("kindFilter").value;
    const state = $("stateFilter").value;
    const rows = allJobs.filter(
      (item) => (!kind || item.kind === kind) && (!state || item.state === state),
    );
    $("jobs").innerHTML = rows.map((item) => `
      <tr data-id="${escapeHtml(item.job_id)}" class="${item.job_id === selectedJobId ? "selected" : ""}">
        <td><b>${escapeHtml(item.name)}</b><small>${escapeHtml(item.message || "")}</small></td>
        <td><span class="pill">${escapeHtml(item.kind)}</span></td>
        <td><span class="state state-${escapeHtml(item.state)}">${escapeHtml(item.state)}</span></td>
        <td><div class="progress"><i style="width:${Math.max(0, Math.min(100, percent(item.progress)))}%"></i></div><small>${percent(item.progress)}%</small></td>
        <td>${item.metrics?.length || 0}</td>
        <td>${item.artifacts?.length || 0}</td>
        <td>${escapeHtml(item.updated_at)}</td>
      </tr>
    `).join("");
    document.querySelectorAll("#jobs tr").forEach((row) => {
      row.onclick = () => selectJob(row.dataset.id);
    });
  }

  const optionalNumber = (id) => {
    const raw = $(id).value.trim();
    if (!raw) return null;
    const value = Number(raw);
    return Number.isFinite(value) ? value : null;
  };

  function promotionPolicy() {
    const mapping = [
      ["min_sharpe", "minSharpe"],
      ["max_drawdown", "maxDrawdown"],
      ["min_reward", "minReward"],
      ["max_loss", "maxLoss"],
      ["min_profit", "minProfit"],
    ];
    const payload = {require_model_files: true};
    mapping.forEach(([key, id]) => {
      const value = optionalNumber(id);
      if (value != null) payload[key] = value;
    });
    return payload;
  }

  function pipelinePromotionPolicy() {
    const mapping = [
      ["min_sharpe", "pipelineMinSharpe"],
      ["max_drawdown", "pipelineMaxDrawdown"],
      ["min_reward", "pipelineMinReward"],
      ["max_loss", "pipelineMaxLoss"],
      ["min_profit", "pipelineMinProfit"],
    ];
    const payload = {require_model_files: true};
    mapping.forEach(([key, id]) => {
      const value = optionalNumber(id);
      if (value != null) payload[key] = value;
    });
    return payload;
  }

  function pipelinePayload() {
    const dayParts = $("pipelineWfDays").value.split("/").map((item) => Number(item.trim()));
    if (dayParts.length !== 3 || dayParts.some((item) => !Number.isFinite(item) || item < 1)) {
      throw new Error("Train / Eval / Step 必须类似 60/15/15");
    }
    const trainingParameters = JSON.parse($("pipelineTrainingParameters").value.trim() || "{}");
    trainingParameters.training_device = $("pipelineTrainingDevice").value;
    trainingParameters.cpu_threads = Math.max(1, Number($("pipelineCpuThreads").value) || 4);
    trainingParameters.gpu_device = Math.max(0, Number($("gpuDevice").value) || 0);
    trainingParameters.min_gpu_free_mb = Math.max(0, Number($("minGpuFreeMb").value) || 0);
    trainingParameters.activate_tensorboard = $("tensorboardEnabled").checked;
    return {
      name: $("pipelineName").value.trim(),
      config_path: $("pipelineConfigPath").value.trim(),
      strategy: $("pipelineStrategy").value.trim(),
      optimization_timerange: $("pipelineOptRange").value.trim(),
      oos_timerange: $("pipelineOosRange").value.trim(),
      training_kind: $("pipelineTrainingKind").value,
      training_device: $("pipelineTrainingDevice").value,
      cpu_threads: Math.max(1, Number($("pipelineCpuThreads").value) || 4),
      walk_forward_start: $("pipelineWfStart").value.trim(),
      walk_forward_end: $("pipelineWfEnd").value.trim(),
      train_days: dayParts[0],
      eval_days: dayParts[1],
      step_days: dayParts[2],
      trials: Math.max(1, Number($("pipelineTrials").value) || 100),
      workers: Math.max(1, Number($("pipelineWorkers").value) || 1),
      top_n: Math.max(1, Math.min(20, Number($("pipelineTopN").value) || 5)),
      max_folds: Math.max(1, Number($("pipelineMaxFolds").value) || 50),
      expanding: $("pipelineExpanding").checked,
      continual_learning: $("pipelineContinual").checked,
      require_training_approval: $("pipelineRequireApproval").checked,
      priority: Math.max(0, Math.min(100, Number($("priority").value) || 50)),
      max_seconds: Math.max(1, Number($("maxSeconds").value) || 14400),
      max_artifact_bytes: 536870912,
      oos_metric: $("pipelineOosMetric").value.trim() || "auto",
      min_oos_success_ratio: Math.max(0.01, Math.min(1, Number($("pipelineOosSuccessRatio").value) || 1)),
      walk_forward_metric: $("pipelineWfMetric").value.trim() || "sharpe",
      min_walk_forward_success_ratio: Math.max(0.01, Math.min(1, Number($("pipelineSuccessRatio").value) || 1)),
      stability_penalty: Math.max(0, Number($("pipelineStabilityPenalty").value) || 0),
      max_stage_retries: Math.max(0, Math.min(10, Number($("pipelineRetries").value) || 0)),
      training_parameters: trainingParameters,
      tags: $("newTags").value.split(",").map((item) => item.trim()).filter(Boolean),
      promotion_policy: pipelinePromotionPolicy(),
      auto_start: $("pipelineAutoStart").checked,
    };
  }

  async function createPipeline() {
    try {
      const payload = await api("/hedge/research/pipelines", {
        method: "POST",
        body: JSON.stringify(pipelinePayload()),
      });
      selectedPipelineId = payload.pipeline_id;
      $("pipelineStatus").textContent = `已创建 ${payload.pipeline_id}`;
      await refreshPipelines();
      await loadPipeline(payload.pipeline_id);
    } catch (error) {
      $("pipelineStatus").textContent = "Pipeline 创建失败: " + error.message;
    }
  }

  function renderPipelines() {
    $("pipelines").innerHTML = allPipelines.map((item) => `
      <div class="pipeline-card ${item.pipeline_id === selectedPipelineId ? "selected" : ""}" data-pipeline-id="${escapeHtml(item.pipeline_id)}">
        <b>${escapeHtml(item.spec?.name || item.pipeline_id)}</b>
        <small>${escapeHtml(item.state)} · ${escapeHtml(item.stage)} · ${percent(item.progress)}%</small>
        <div class="pipeline-progress"><i style="width:${Math.max(0, Math.min(100, percent(item.progress)))}%"></i></div>
        <small>${escapeHtml(item.message || "")}</small>
      </div>
    `).join("") || '<div class="pipeline-card">暂无 Pipeline。</div>';
    $("pipelines").querySelectorAll("[data-pipeline-id]").forEach((row) => {
      row.onclick = () => loadPipeline(row.dataset.pipelineId);
    });
  }

  function pipelineNodeHtml(node) {
    const progressValue = Math.max(0, Math.min(100, percent(node.progress || 0)));
    return `<div class="pipeline-node state-${escapeHtml(node.state || "PENDING")}">
      <b>${escapeHtml(node.label || node.id)}</b>
      <small>${escapeHtml(node.state || "PENDING")}${node.kind ? " · " + escapeHtml(node.kind) : ""}</small>
      ${node.job_id ? `<small>${escapeHtml(node.job_id.slice(0, 18))}</small>` : ""}
      <div class="pipeline-progress"><i style="width:${progressValue}%"></i></div>
    </div>`;
  }

  async function pipelineAction(pipelineId, action) {
    try {
      await api(`/hedge/research/pipelines/${encodeURIComponent(pipelineId)}/${action}`, {method: "POST"});
      await refreshPipelines();
      await loadPipeline(pipelineId);
    } catch (error) {
      $("pipelineStatus").textContent = `${action} 失败: ${error.message}`;
    }
  }

  async function loadPipeline(pipelineId) {
    selectedPipelineId = pipelineId;
    try {
      const item = await api(`/hedge/research/pipelines/${encodeURIComponent(pipelineId)}`);
      const nodes = item.dag?.nodes || [];
      const edges = item.dag?.edges || [];
      const eventRows = (item.events || []).slice(-20).reverse().map(
        (event) => `<div>${escapeHtml(event.at || "")} · <b>${escapeHtml(event.event || "")}</b> · ${escapeHtml(event.message || "")}</div>`,
      ).join("");
      const selected = item.oos?.selected || {};
      const aggregate = item.walk_forward?.aggregate_metrics || {};
      const promotion = item.promotion || {};
      const canStart = item.state === "QUEUED";
      const canPause = item.state === "RUNNING";
      const canResume = item.state === "PAUSED";
      const canCancel = !["SUCCEEDED", "REJECTED", "FAILED", "CANCELED"].includes(item.state);
      const canRetry = item.state === "FAILED";
      const canReconsider = item.state === "REJECTED" && Boolean(item.promotion?.candidate);
      const canApprove = item.state === "AWAITING_APPROVAL";
      $("pipelineDetail").innerHTML = `
        <div class="section-title"><div><h3>${escapeHtml(item.spec?.name || item.pipeline_id)}</h3><p>${escapeHtml(item.pipeline_id)}</p></div><span class="state state-${escapeHtml(item.state)}">${escapeHtml(item.state)}</span></div>
        <div class="pipeline-progress"><i style="width:${Math.max(0, Math.min(100, percent(item.progress)))}%"></i></div>
        <div class="pipeline-actions">
          <button data-pipe-action="start" ${canStart ? "" : "disabled"}>启动</button>
          <button data-pipe-action="pause" ${canPause ? "" : "disabled"}>暂停</button>
          <button data-pipe-action="resume" ${canResume ? "" : "disabled"}>继续</button>
          <button data-pipe-action="approve-training" ${canApprove ? "" : "disabled"}>批准训练</button>
          <button data-pipe-action="cancel" ${canCancel ? "" : "disabled"}>取消</button>
          <button data-pipe-action="retry" ${canRetry ? "" : "disabled"}>重试失败 Stage</button>
          <button data-pipe-action="reconsider" ${canReconsider ? "" : "disabled"}>用当前 Gate 重审</button>
        </div>
        <div class="runtime-strip">
          <span>Stage <b>${escapeHtml(item.stage)}</b></span>
          <span>OOS Trial <b>${escapeHtml(selected.trial_id ?? "-")}</b></span>
          <span>OOS Metric <b>${escapeHtml(item.oos?.ranking_metric || "-")}</b></span>
          <span>WF Success <b>${((Number(item.walk_forward?.success_ratio || 0)) * 100).toFixed(0)}%</b></span>
          <span>Candidate <b>${escapeHtml(promotion.target || "-")}</b></span>
        </div>
        <h3>DAG</h3>
        <div class="pipeline-dag">${nodes.map((node, index) => `${index ? '<div class="pipeline-edge">→</div>' : ""}${pipelineNodeHtml(node)}`).join("")}</div>
        <small>依赖边 ${edges.length}；OOS fan-out 与 continual fold 依赖均来自后端真实 DAG。</small>
        <h3>Walk-forward 聚合</h3><pre class="compact-pre">${escapeHtml(JSON.stringify(aggregate, null, 2))}</pre>
        <h3>Promotion Gate</h3><pre class="compact-pre">${escapeHtml(JSON.stringify(promotion.gate || {}, null, 2))}</pre>
        <h3>最近事件</h3><div class="pipeline-events">${eventRows || "暂无事件"}</div>
      `;
      $("pipelineDetail").querySelectorAll("[data-pipe-action]").forEach((button) => {
        button.onclick = async () => {
          if (button.dataset.pipeAction === "reconsider") {
            try {
              await api(`/hedge/research/pipelines/${encodeURIComponent(pipelineId)}/reconsider-promotion`, {
                method: "POST",
                body: JSON.stringify(pipelinePromotionPolicy()),
              });
              await refreshPipelines();
              await loadPipeline(pipelineId);
            } catch (error) {
              $("pipelineStatus").textContent = "重新评估 Promotion 失败: " + error.message;
            }
            return;
          }
          await pipelineAction(pipelineId, button.dataset.pipeAction);
        };
      });
      renderPipelines();
      $("pipelineStatus").textContent = `${item.state} · ${item.stage} · ${percent(item.progress)}%`;
    } catch (error) {
      $("pipelineStatus").textContent = "Pipeline 读取失败: " + error.message;
    }
  }

  async function refreshPipelines() {
    try {
      const payload = await api("/hedge/research/pipelines?limit=100");
      allPipelines = payload.pipelines || [];
      renderPipelines();
      if (selectedPipelineId && allPipelines.some((item) => item.pipeline_id === selectedPipelineId)) {
        await loadPipeline(selectedPipelineId);
      }
    } catch (error) {
      $("pipelineStatus").textContent = "Pipeline 刷新失败: " + error.message;
    }
  }

  function renderWalkForwardGroups(rows) {
    $("wfGroups").innerHTML = rows.map((item) => {
      const states = item.job_states || {};
      const metrics = item.metrics || {};
      const metricText = Object.entries(metrics).slice(0, 4).map(
        ([name, row]) => `${name} mean=${Number(row.mean).toFixed(4)} min=${Number(row.min).toFixed(4)}`,
      ).join(" · ");
      return `<div class="wf-group" data-wf-group="${escapeHtml(item.group_id)}">
        <b>${escapeHtml(item.name || item.group_id)}</b>
        <small>${escapeHtml(item.group_id)} · success ${(Number(item.success_ratio || 0) * 100).toFixed(0)}% · ${JSON.stringify(states)}</small>
        <div class="wf-metrics">${escapeHtml(metricText || "等待指标")}</div>
      </div>`;
    }).join("") || '<div class="wf-group">暂无 walk-forward 批任务。</div>';
    $("wfGroups").querySelectorAll("[data-wf-group]").forEach((row) => {
      row.onclick = async () => {
        try {
          const payload = await api(`/hedge/research/walk-forward/${encodeURIComponent(row.dataset.wfGroup)}`);
          $("wfPreviewOutput").textContent = JSON.stringify(payload, null, 2);
          $("wfStatus").textContent = `${payload.group_id} · success ${(Number(payload.success_ratio || 0) * 100).toFixed(0)}%`;
        } catch (error) {
          $("wfStatus").textContent = "批任务读取失败: " + error.message;
        }
      };
    });
  }

  async function refreshCatalog() {
    try {
      const metric = $("leaderboardMetric").value;
      const maximize = $("leaderboardMaximize").checked;
      const [board, models, promotions, groups] = await Promise.all([
        api(`/hedge/research/experiments/leaderboard?metric=${encodeURIComponent(metric)}&maximize=${maximize}&limit=100`),
        api("/hedge/research/models?limit=100"),
        api("/hedge/research/promotions?limit=100"),
        api("/hedge/research/walk-forward?limit=50"),
      ]);
      renderExperiments(board.rows || []);
      renderModels(models.models || []);
      renderPromotions(promotions.promotions || []);
      renderWalkForwardGroups(groups.groups || []);
    } catch (error) {
      $("experimentStatus").textContent = "实验目录读取失败: " + error.message;
    }
  }

  function renderExperiments(rows) {
    $("experiments").innerHTML = rows.map((item) => `
      <tr>
        <td><b>${escapeHtml(item.name)}</b><small>${escapeHtml(item.timerange || "")}</small></td>
        <td><span class="pill">${escapeHtml(item.kind)}</span></td>
        <td><small>${escapeHtml(item.identifier || "-")}</small></td>
        <td>${Number(item.leaderboard_score || 0).toFixed(5)}</td>
        <td>${item.model_files?.length || 0} · ${formatBytes(item.model_bytes)}</td>
        <td><div class="mini-actions">
          <button data-resume="${escapeHtml(item.experiment_id)}" ${!["ML_TRAIN", "RL_TRAIN"].includes(item.kind) ? "disabled" : ""}>续训</button>
          <button data-tensorboard="${escapeHtml(item.experiment_id)}">训练曲线</button>
          <button data-promote="${escapeHtml(item.experiment_id)}">晋级 dry-run</button>
        </div></td>
      </tr>
    `).join("") || '<tr><td colspan="6">当前指标暂无可排名实验。</td></tr>';
    $("experiments").querySelectorAll("[data-resume]").forEach((button) => {
      button.onclick = async () => {
        try {
          const payload = await api(`/hedge/research/experiments/${encodeURIComponent(button.dataset.resume)}/resume`, {
            method: "POST",
            body: JSON.stringify({auto_execute: true}),
          });
          $("experimentStatus").textContent = "已创建续训任务 " + payload.job_id;
          await refresh();
          await selectJob(payload.job_id);
        } catch (error) {
          $("experimentStatus").textContent = "续训创建失败: " + error.message;
        }
      };
    });
    $("experiments").querySelectorAll("[data-tensorboard]").forEach((button) => {
      button.onclick = async () => {
        try {
          const payload = await api(`/hedge/research/experiments/${encodeURIComponent(button.dataset.tensorboard)}/tensorboard?max_points=1200&max_tags=100`);
          if (!payload.available) {
            $("experimentStatus").textContent = payload.message || "TensorBoard reader unavailable";
            return;
          }
          renderTensorboard(payload.tags || {});
          $("experimentStatus").textContent = `TensorBoard: ${payload.tag_count || 0} scalar tags`;
        } catch (error) {
          $("experimentStatus").textContent = "TensorBoard 读取失败: " + error.message;
        }
      };
    });
    $("experiments").querySelectorAll("[data-promote]").forEach((button) => {
      button.onclick = async () => {
        try {
          const payload = await api(`/hedge/research/experiments/${encodeURIComponent(button.dataset.promote)}/promote`, {
            method: "POST",
            body: JSON.stringify(promotionPolicy()),
          });
          $("experimentStatus").textContent = `已晋级 ${payload.promotion_id}；dry-run override: ${payload.dry_run_override_path}`;
        } catch (error) {
          $("experimentStatus").textContent = "晋级失败: " + error.message;
        }
      };
    });
  }

  function drawSeriesCanvas(canvas, rows, label) {
    const width = Math.max(520, canvas.parentElement.clientWidth - 20);
    const height = 190;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);
    if (!rows.length) return;
    const values = rows.map((item) => Number(item.value));
    let min = Math.min(...values);
    let max = Math.max(...values);
    if (max === min) { max += 1; min -= 1; }
    const pad = 28;
    ctx.strokeStyle = "#243b57";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, 12); ctx.lineTo(pad, height - pad); ctx.lineTo(width - 8, height - pad); ctx.stroke();
    ctx.strokeStyle = "#63e6be";
    ctx.lineWidth = 2;
    ctx.beginPath();
    rows.forEach((item, index) => {
      const x = pad + (rows.length === 1 ? 0 : index * (width - pad - 12) / (rows.length - 1));
      const y = height - pad - ((Number(item.value) - min) / (max - min)) * (height - pad * 1.5);
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = "#9cb0c8";
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillText(`${label}: ${values[values.length - 1].toFixed(6)}`, pad + 6, 16);
  }

  function renderTensorboard(tags) {
    const names = Object.keys(tags).sort();
    const select = $("tbTag");
    select.innerHTML = names.map((name) => `<option>${escapeHtml(name)}</option>`).join("");
    const draw = () => drawSeriesCanvas($("tbChart"), tags[select.value] || [], select.value || "tensorboard");
    select.onchange = draw;
    draw();
  }

  function renderPromotions(rows) {
    $("promotions").innerHTML = rows.map((item) => `<div class="model-row">
      <b>${escapeHtml(item.identifier || item.promotion_id)}</b>
      <small>${escapeHtml(item.promotion_id)} · ${escapeHtml(item.created_at || "")}</small>
      <div class="model-files">${escapeHtml(item.target || "DRY_RUN_CANDIDATE")} · live write disabled</div>
    </div>`).join("") || '<div class="model-row">暂无已晋级 dry-run candidate。</div>';
  }

  function renderModels(rows) {
    $("modelSummary").textContent = `${rows.length} identifiers`;
    $("models").innerHTML = rows.map((item) => {
      const roles = {};
      (item.files || []).forEach((file) => { roles[file.role] = (roles[file.role] || 0) + 1; });
      return `<div class="model-row">
        <b>${escapeHtml(item.identifier)}</b>
        <small>${item.file_count} files · ${formatBytes(item.bytes)} · ${escapeHtml(item.latest_modified_at || "")}</small>
        <div class="model-files">best ${roles.best || 0} · checkpoint ${roles.checkpoint || 0} · model ${roles.model || 0}</div>
      </div>`;
    }).join("") || '<div class="model-row">尚未发现 FreqAI 模型目录。</div>';
  }

  function walkForwardPayload(includeTemplate) {
    const payload = {
      start: $("wfStart").value.trim(),
      end: $("wfEnd").value.trim(),
      train_days: Math.max(1, Number($("wfTrainDays").value) || 1),
      eval_days: Math.max(1, Number($("wfEvalDays").value) || 1),
      step_days: Math.max(1, Number($("wfStepDays").value) || 1),
      expanding: $("wfExpanding").checked,
      max_folds: Math.max(1, Number($("wfMaxFolds").value) || 50),
    };
    if (includeTemplate) {
      payload.kind = $("newKind").value;
      payload.name = $("newName").value.trim() || "walk-forward";
      payload.parameters = buildParameters();
      payload.parameters.walk_forward_continual = $("wfContinual").checked;
      payload.tags = $("newTags").value.split(",").map((item) => item.trim()).filter(Boolean);
      payload.priority = Math.max(0, Math.min(100, Number($("priority").value) || 0));
      payload.auto_execute = $("wfAutoExecute").checked;
      payload.budget = {
        max_seconds: Math.max(1, Number($("maxSeconds").value) || 3600),
        max_trials: Math.max(1, Number($("trials").value) || 100),
        max_workers: Math.max(1, Number($("workers").value) || 1),
        max_artifact_bytes: 268435456,
      };
    }
    return payload;
  }

  async function previewWalkForward() {
    try {
      const payload = await api("/hedge/research/walk-forward/plan", {
        method: "POST",
        body: JSON.stringify(walkForwardPayload(false)),
      });
      $("wfPreviewOutput").textContent = JSON.stringify(payload, null, 2);
      $("wfStatus").textContent = `${payload.fold_count} folds`;
    } catch (error) {
      $("wfStatus").textContent = "预览失败: " + error.message;
    }
  }

  async function submitWalkForward() {
    try {
      const payload = await api("/hedge/research/walk-forward/submit", {
        method: "POST",
        body: JSON.stringify(walkForwardPayload(true)),
      });
      $("wfPreviewOutput").textContent = JSON.stringify(payload.group, null, 2);
      $("wfStatus").textContent = `已创建 ${payload.jobs.length} 个 fold · ${payload.group.group_id}`;
      await refresh();
    } catch (error) {
      $("wfStatus").textContent = "创建失败: " + error.message;
    }
  }

  async function loadLog(jobId) {
    try {
      const payload = await api(`/hedge/research/jobs/${encodeURIComponent(jobId)}/log?lines=400`);
      $("liveLog").textContent = (payload.lines || []).join("\n") || "暂无进程输出。";
      $("liveLog").scrollTop = $("liveLog").scrollHeight;
      $("logStatus").textContent = `${payload.line_count || 0} 行 · ${new Date().toLocaleTimeString()}`;
    } catch (error) {
      $("logStatus").textContent = "日志读取失败: " + error.message;
    }
  }

  function drawMetricChart(jobId, metrics) {
    const canvas = $("metricChart");
    const select = $("metricSelect");
    if (!canvas || !select) return;
    const names = [...new Set((metrics || []).map((item) => item.name))];
    const priority = ["reward", "loss", "sharpe", "profit", "drawdown", "accuracy", "f1", "rmse", "mae"];
    const preferred = selectedMetric[jobId] || priority.find((name) => names.includes(name)) || names[0] || "";
    select.innerHTML = names.map((name) => `<option ${name === preferred ? "selected" : ""}>${escapeHtml(name)}</option>`).join("");
    select.disabled = names.length === 0;
    if (!preferred) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    selectedMetric[jobId] = preferred;
    const rows = metrics.filter((item) => item.name === preferred);
    const width = Math.max(520, canvas.parentElement.clientWidth - 20);
    const height = 190;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);
    if (!rows.length) return;
    const values = rows.map((item) => Number(item.value));
    let min = Math.min(...values);
    let max = Math.max(...values);
    if (max === min) { max += 1; min -= 1; }
    const pad = 28;
    ctx.strokeStyle = "#243b57";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, pad / 2);
    ctx.lineTo(pad, height - pad);
    ctx.lineTo(width - 8, height - pad);
    ctx.stroke();
    ctx.strokeStyle = preferred === "loss" || preferred === "drawdown" ? "#fb7185" : "#63e6be";
    ctx.lineWidth = 2;
    ctx.beginPath();
    rows.forEach((item, index) => {
      const x = pad + (rows.length === 1 ? 0 : index * (width - pad - 12) / (rows.length - 1));
      const y = height - pad - ((Number(item.value) - min) / (max - min)) * (height - pad * 1.5);
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = "#9cb0c8";
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillText(`${preferred}: ${values[values.length - 1].toFixed(6)}`, pad + 6, 16);
    ctx.fillText(`min ${min.toFixed(4)}`, pad + 6, height - 8);
    ctx.fillText(`max ${max.toFixed(4)}`, width - 110, 16);
    select.onchange = () => {
      selectedMetric[jobId] = select.value;
      drawMetricChart(jobId, metrics);
    };
  }

  async function loadDetail(jobId) {
    const item = await api("/hedge/research/jobs/" + encodeURIComponent(jobId));
    const runtime = item.runtime || {};
    const artifacts = (item.artifacts || []).map((artifact) => {
      const path = artifact.relative_path.split("/").map(encodeURIComponent).join("/");
      const href = `/api/v1/hedge/research/jobs/${encodeURIComponent(jobId)}/artifacts/${path}`;
      return `<li><a href="#" data-artifact="${escapeHtml(href)}" data-name="${escapeHtml(artifact.name)}">${escapeHtml(artifact.name)}</a> · ${formatBytes(artifact.size)}</li>`;
    }).join("");
    const processResource = runtime.resources || {};
    let replayInfo = null;
    if (item.kind === "OPTIMIZATION") {
      try {
        replayInfo = await api(`/hedge/research/jobs/${encodeURIComponent(jobId)}/optimization-replays`);
      } catch (_) {
        replayInfo = null;
      }
    }
    const replayRows = (replayInfo?.oos_leaderboard || []).map((row) => `
      <tr><td>${row.rank}</td><td>${escapeHtml(row.trial_id ?? "-")}</td><td>${escapeHtml(row.state || "")}</td><td>${Number(row.value || 0).toFixed(6)}</td></tr>`
    ).join("");
    const replaySection = item.kind === "OPTIMIZATION" ? `
      <h3>Optimization → OOS Replay</h3>
      <div class="runtime-strip"><span>Ranking <b>${escapeHtml(replayInfo?.ranking_metric || "等待 OOS 指标")}</b></span><span>Replays <b>${(replayInfo?.replays || []).length}</b></span></div>
      <div class="table-wrap"><table><thead><tr><th>OOS Rank</th><th>Trial</th><th>State</th><th>Metric</th></tr></thead><tbody>${replayRows || '<tr><td colspan="4">暂无 OOS replay</td></tr>'}</tbody></table></div>
    ` : "";
    $("detail").innerHTML = `
      <div class="detail-actions">
        <button data-action="execute" ${item.state !== "QUEUED" ? "disabled" : ""}>执行</button>
        <button data-action="pause" ${item.state !== "RUNNING" ? "disabled" : ""}>暂停进程</button>
        <button data-action="resume" ${item.state !== "PAUSED" ? "disabled" : ""}>继续进程</button>
        <button data-action="cancel" ${["SUCCEEDED", "FAILED", "CANCELED"].includes(item.state) ? "disabled" : ""}>取消</button>
        <button data-action="retry" ${!["FAILED", "CANCELED"].includes(item.state) ? "disabled" : ""}>重试</button>
        <button data-action="replay-best" ${item.kind !== "OPTIMIZATION" || item.state !== "SUCCEEDED" ? "disabled" : ""}>最佳参数回测</button>
        <button data-action="replay-top" ${item.kind !== "OPTIMIZATION" || item.state !== "SUCCEEDED" ? "disabled" : ""}>Top 候选 OOS</button>
        <button data-action="log">刷新日志</button>
      </div>
      <div class="runtime-strip">
        <span>PID <b>${escapeHtml(runtime.pid ?? "-")}</b></span>
        <span>CPU <b>${Number(processResource.cpu_percent || 0).toFixed(1)}%</b></span>
        <span>Train Device <b>${escapeHtml(runtime.training_device || "-")}</b></span>
        <span>CPU Threads <b>${escapeHtml(runtime.cpu_threads ?? "-")}</b></span>
        <span>RSS <b>${formatBytes(processResource.rss_bytes)}</b></span>
        <span>Elapsed <b>${Number(runtime.elapsed_seconds || 0).toFixed(1)}s</b></span>
      </div>
      <div class="metric-head"><h3>训练 / 评估指标</h3><select id="metricSelect"></select></div>
      <canvas id="metricChart" class="metric-chart" width="640" height="190"></canvas>
      ${replaySection}
      <details><summary>原始任务状态</summary><pre>${escapeHtml(JSON.stringify(item, null, 2))}</pre></details>
      <h3>Artifacts</h3><ul>${artifacts || "<li>暂无</li>"}</ul>
    `;
    drawMetricChart(jobId, item.metrics || []);
    const execute = $("detail").querySelector('[data-action="execute"]');
    const pause = $("detail").querySelector('[data-action="pause"]');
    const resume = $("detail").querySelector('[data-action="resume"]');
    const cancel = $("detail").querySelector('[data-action="cancel"]');
    const retry = $("detail").querySelector('[data-action="retry"]');
    const replayBest = $("detail").querySelector('[data-action="replay-best"]');
    const replayTop = $("detail").querySelector('[data-action="replay-top"]');
    execute.onclick = async () => {
      await api(`/hedge/research/jobs/${encodeURIComponent(jobId)}/execute`, {method: "POST"});
      await refresh();
      await loadDetail(jobId);
    };
    pause.onclick = async () => {
      await api(`/hedge/research/jobs/${encodeURIComponent(jobId)}/pause`, {method: "POST"});
      await refresh();
      await loadDetail(jobId);
    };
    resume.onclick = async () => {
      await api(`/hedge/research/jobs/${encodeURIComponent(jobId)}/resume`, {method: "POST"});
      await refresh();
      await loadDetail(jobId);
    };
    cancel.onclick = async () => {
      await api(`/hedge/research/jobs/${encodeURIComponent(jobId)}/cancel`, {method: "POST"});
      await refresh();
      await loadDetail(jobId);
    };
    retry.onclick = async () => {
      const retried = await api(`/hedge/research/jobs/${encodeURIComponent(jobId)}/retry`, {method: "POST"});
      await refresh();
      await selectJob(retried.job_id);
    };
    replayBest.onclick = async () => {
      const requested = window.prompt("最佳参数回测 Timerange（留空则使用原任务 timerange）", "") ?? "";
      const payload = await api(`/hedge/research/jobs/${encodeURIComponent(jobId)}/replay-best`, {
        method: "POST",
        body: JSON.stringify({timerange: requested.trim(), auto_execute: true}),
      });
      const childId = payload.replay_job?.job_id;
      await refresh();
      if (childId) await selectJob(childId);
    };
    replayTop.onclick = async () => {
      const count = Math.max(1, Math.min(20, Number(window.prompt("回测前多少个候选？", "5")) || 5));
      const requested = window.prompt("Top 候选 OOS Timerange", "") ?? "";
      const payload = await api(`/hedge/research/jobs/${encodeURIComponent(jobId)}/replay-top`, {
        method: "POST",
        body: JSON.stringify({limit: count, timerange: requested.trim(), auto_execute: true}),
      });
      await refresh();
      if (payload.jobs?.length) await selectJob(payload.jobs[0].job_id);
    };
    $("detail").querySelector('[data-action="log"]').onclick = () => loadLog(jobId);
    $("detail").querySelectorAll("[data-artifact]").forEach((link) => {
      link.onclick = async (event) => {
        event.preventDefault();
        const response = await fetch(link.dataset.artifact, {
          headers: auth ? {Authorization: auth} : {},
        });
        if (!response.ok) throw new Error(await response.text());
        const blobUrl = URL.createObjectURL(await response.blob());
        const download = document.createElement("a");
        download.href = blobUrl;
        download.download = link.dataset.name;
        download.click();
        URL.revokeObjectURL(blobUrl);
      };
    });
  }

  async function selectJob(jobId) {
    selectedJobId = jobId;
    renderJobs();
    await Promise.all([loadDetail(jobId), loadLog(jobId)]);
  }

  function buildParameters() {
    const raw = $("newParameters").value.trim() || "{}";
    const parameters = JSON.parse(raw);
    parameters.config_path = $("configPath").value.trim();
    parameters.strategy = $("strategy").value.trim();
    const timerange = $("timerange").value.trim();
    if (timerange) parameters.timerange = timerange;
    const trials = Number($("trials").value);
    const workers = Number($("workers").value);
    if (Number.isFinite(trials) && trials > 0) parameters.trials = trials;
    if (Number.isFinite(workers) && workers > 0) parameters.workers = workers;
    parameters.auto_replay_best = $("autoReplayBest").checked;
    parameters.replay_top_n = Math.max(1, Math.min(20, Number($("replayTopN").value) || 1));
    const replayTimerange = $("replayTimerange").value.trim();
    if (replayTimerange) parameters.replay_timerange = replayTimerange;
    const identifier = $("freqaiIdentifier").value.trim();
    if (identifier) parameters.freqai_identifier = identifier;
    const trainDays = Number($("trainPeriodDays").value);
    const evalDays = Number($("backtestPeriodDays").value);
    if (Number.isFinite(trainDays) && trainDays > 0) parameters.train_period_days = trainDays;
    if (Number.isFinite(evalDays) && evalDays > 0) parameters.backtest_period_days = evalDays;
    parameters.training_device = $("trainingDevice").value;
    parameters.cpu_threads = Math.max(1, Number($("cpuThreads").value) || 4);
    parameters.gpu_device = Math.max(0, Number($("gpuDevice").value) || 0);
    parameters.min_gpu_free_mb = Math.max(0, Number($("minGpuFreeMb").value) || 0);
    parameters.activate_tensorboard = $("tensorboardEnabled").checked;
    return parameters;
  }

  async function createJob() {
    try {
      const tags = $("newTags").value.split(",").map((item) => item.trim()).filter(Boolean);
      const payload = {
        kind: $("newKind").value,
        name: $("newName").value.trim(),
        tags,
        parameters: buildParameters(),
        auto_execute: $("autoExecute").checked,
        priority: Math.max(0, Math.min(100, Number($("priority").value) || 0)),
        budget: {
          max_seconds: Math.max(1, Number($("maxSeconds").value) || 3600),
          max_trials: Math.max(1, Number($("trials").value) || 100),
          max_workers: Math.max(1, Number($("workers").value) || 1),
          max_artifact_bytes: 268435456,
        },
      };
      const job = await api("/hedge/research/jobs", {method: "POST", body: JSON.stringify(payload)});
      $("createStatus").textContent = "已创建 " + job.job_id;
      await refresh();
      await selectJob(job.job_id);
    } catch (error) {
      $("createStatus").textContent = "创建失败: " + error.message;
    }
  }

  async function refresh() {
    try {
      const [dashboard, jobs, capabilities] = await Promise.all([
        api("/hedge/research/dashboard"),
        api("/hedge/research/jobs?limit=500"),
        api("/hedge/research/capabilities"),
      ]);
      allJobs = jobs.jobs || [];
      renderCards(dashboard);
      renderResources(dashboard.executor || {});
      renderJobs();
      await Promise.all([refreshCatalog(), refreshPipelines()]);
      $("capabilities").textContent = JSON.stringify(capabilities, null, 2);
      $("status").textContent = "已连接 · " + new Date().toLocaleTimeString();
      if (selectedJobId) {
        await Promise.all([loadDetail(selectedJobId), loadLog(selectedJobId)]);
      }
    } catch (error) {
      $("status").textContent = "错误: " + error.message;
    }
  }

  $("connect").onclick = () => {
    auth = "Basic " + btoa($("user").value + ":" + $("password").value);
    refresh();
    clearInterval(timer);
    timer = setInterval(refresh, 3000);
  };
  $("refresh").onclick = refresh;
  $("createJob").onclick = createJob;
  $("createPipeline").onclick = createPipeline;
  $("refreshPipelines").onclick = refreshPipelines;
  $("wfPreview").onclick = previewWalkForward;
  $("wfSubmit").onclick = submitWalkForward;
  $("refreshExperiments").onclick = refreshCatalog;
  $("leaderboardMetric").onchange = () => {
    $("leaderboardMaximize").checked = !["loss", "drawdown"].includes($("leaderboardMetric").value);
    refreshCatalog();
  };
  $("leaderboardMaximize").onchange = refreshCatalog;
  $("pauseQueue").onclick = async () => { await api("/hedge/research/executor/pause", {method: "POST"}); await refresh(); };
  $("resumeQueue").onclick = async () => { await api("/hedge/research/executor/resume", {method: "POST"}); await refresh(); };
  $("kindFilter").onchange = renderJobs;
  $("stateFilter").onchange = renderJobs;
})();
