"""Optional loopback-only Hedge dashboard API and embedded UI."""
from __future__ import annotations
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from freqtrade.hedge.control.auth import HedgePrincipal, HedgeRole
from freqtrade.rpc.api_server.hedge_auth import require_role
from freqtrade.rpc.api_server.hedge_dashboard_schemas import (
    DashboardControlRequest,DashboardControlSchema,DashboardCycleSchema,DashboardCyclesSchema,
    DashboardDefaultsSchema,DashboardOverviewSchema,DashboardStrategySchema,
)

_SECURITY={"Cache-Control":"no-store, max-age=0","Pragma":"no-cache","X-Content-Type-Options":"nosniff","X-Frame-Options":"DENY","Referrer-Policy":"no-referrer","Content-Security-Policy":"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"}

def _bot():
    from freqtrade.rpc.api_server.deps import get_rpc_optional
    rpc=get_rpc_optional();return getattr(rpc,"_freqtrade",None)

def _cycle(item)->DashboardCycleSchema:
    return DashboardCycleSchema(strategy=DashboardStrategySchema(**asdict(item.strategy)),**{k:v for k,v in asdict(item).items() if k not in {"account_id","symbol","strategy"}})

class HedgeDashboardQuery:
    def __init__(self, *, account_id:str, symbol:str, refresh_seconds:int=5, bot_provider:Callable[[],Any]|None=None)->None:
        self.account_id=account_id;self.symbol=symbol;self.refresh_seconds=refresh_seconds;self._bot_provider=bot_provider or _bot
    def _application(self):return getattr(self._bot_provider(),"hedge_application",None)
    def defaults(self)->DashboardDefaultsSchema:return DashboardDefaultsSchema(account_id=self.account_id,symbol=self.symbol,refresh_seconds=self.refresh_seconds,local_only=True)
    def cycles(self,limit:int=200)->DashboardCyclesSchema:
        app=self._application();store=getattr(app,"telemetry",None);items=() if store is None else store.list(limit)
        rows=tuple(_cycle(x) for x in items);return DashboardCyclesSchema(cycles=rows,count=len(rows))
    def overview(self)->DashboardOverviewSchema:
        app=self._application();store=getattr(app,"telemetry",None);control=getattr(app,"dryrun_control",None)
        latest=None if store is None or store.latest() is None else _cycle(store.latest())
        control_schema=None
        if control is not None:
            snap=control.snapshot();control_schema=DashboardControlSchema(mode=snap.mode.value,revision=snap.revision,updated_at=snap.updated_at,actor=snap.actor,reason=snap.reason,new_risk_enabled=snap.new_risk_enabled)
        readiness_state="UNAVAILABLE";reasons=()
        runtime=getattr(self._bot_provider(),"hedge_runtime",None)
        if runtime is not None:
            try:
                view=runtime.view();readiness_state="READY" if view.ready else "NOT_READY";reasons=tuple(view.reasons)
            except Exception as exc:
                readiness_state="NOT_READY";reasons=(f"READINESS_ERROR:{type(exc).__name__}",)
        orders=()
        execution=getattr(app,"execution",None)
        if execution is not None:
            try:
                rows=execution.core.list_orders(account_id=self.account_id,symbol=None,include_terminal=False)
                orders=tuple({"client_order_id":x.client_order_id,"symbol":x.intent.symbol,"position_side":x.intent.position_side.value,"action":x.intent.action.value,"status":x.lifecycle.status.value,"filled_quantity":str(x.lifecycle.filled_quantity),"approved_quantity":str(x.approved_quantity),"limit_price":None if x.intent.limit_price is None else str(x.intent.limit_price)} for x in rows)
            except Exception: orders=()
        operations_runtime=getattr(app,"operations",None)
        operations_snapshot=None if operations_runtime is None or operations_runtime.latest is None else operations_runtime.latest.summary()
        return DashboardOverviewSchema(account_id=self.account_id,symbol=self.symbol,mode="DRY_RUN",readiness_state=readiness_state,readiness_reasons=reasons,control=control_schema,latest=latest,active_orders=orders,telemetry_error=None if store is None else store.last_error,operations=operations_snapshot,operations_error=getattr(app,"operations_error",None),generated_at=datetime.now(UTC))
    def control(self, request:DashboardControlRequest, principal:HedgePrincipal)->DashboardControlSchema:
        app=self._application();state=getattr(app,"dryrun_control",None)
        if state is None:raise RuntimeError("Dry-run control state is unavailable")
        if request.action=="pause":snap=state.pause_new_risk(actor=principal.subject,reason=request.reason or "DASHBOARD_PAUSE")
        elif request.action=="resume":snap=state.resume_new_risk(actor=principal.subject,reason=request.reason or "DASHBOARD_RESUME")
        else:snap=state.reset_fail_closed(actor=principal.subject)
        return DashboardControlSchema(mode=snap.mode.value,revision=snap.revision,updated_at=snap.updated_at,actor=snap.actor,reason=snap.reason,new_risk_enabled=snap.new_risk_enabled)

def create_hedge_dashboard_router(*,query:HedgeDashboardQuery,principal_dependency:Callable[...,HedgePrincipal])->APIRouter:
    router=APIRouter(tags=["hedge-dashboard"]);viewer=require_role(HedgeRole.VIEWER,principal_dependency);operator=require_role(HedgeRole.OPERATOR,principal_dependency)
    @router.get("/hedge/dashboard/defaults",response_model=DashboardDefaultsSchema)
    def defaults(_:HedgePrincipal=Depends(viewer)):return query.defaults()
    @router.get("/hedge/dashboard/overview",response_model=DashboardOverviewSchema)
    def overview(_:HedgePrincipal=Depends(viewer)):
        try:return query.overview()
        except RuntimeError as exc:raise HTTPException(status_code=503,detail=str(exc)) from exc
    @router.get("/hedge/dashboard/cycles",response_model=DashboardCyclesSchema)
    def cycles(limit:int=200,_:HedgePrincipal=Depends(viewer)):return query.cycles(max(1,min(limit,2000)))
    @router.get("/hedge/dashboard/strategy-contract")
    def strategy_contract(_:HedgePrincipal=Depends(viewer)):
        from freqtrade.hedge.strategies.contract import StrategyContract
        return asdict(StrategyContract())
    @router.post("/hedge/dry-run/control",response_model=DashboardControlSchema)
    def control(request:DashboardControlRequest,principal:HedgePrincipal=Depends(operator)):
        try:return query.control(request,principal)
        except RuntimeError as exc:raise HTTPException(status_code=503,detail=str(exc)) from exc
    return router

def create_hedge_dashboard_ui_router()->APIRouter:
    router=APIRouter(tags=["hedge-dashboard-ui"]);assets=Path(__file__).with_name("hedge_ui")
    @router.get("/hedge-dashboard",include_in_schema=False)
    def index():return HTMLResponse((assets/"index.html").read_text(encoding="utf-8"),headers=_SECURITY)
    @router.get("/hedge-dashboard/assets/{name}",include_in_schema=False)
    def asset(name:str):
        if name not in {"app.js","styles.css"}:raise HTTPException(status_code=404,detail="asset not found")
        media="application/javascript" if name.endswith(".js") else "text/css"
        return FileResponse(assets/name,media_type=media,headers=_SECURITY)
    return router
