"""Pydantic responses for the optional local Hedge Dry-run dashboard."""
from __future__ import annotations
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

class DashboardStrategySchema(BaseModel):
    model_config=ConfigDict(frozen=True)
    long_score:Decimal=Decimal("0");short_score:Decimal=Decimal("0")
    target_net_quantity:Decimal|None=None;target_net_ratio:Decimal|None=None
    confidence:Decimal=Decimal("1");risk_scale:Decimal=Decimal("1")
    long_exposure_scale:Decimal=Decimal("1");short_exposure_scale:Decimal=Decimal("1")
    allow_new_risk:bool=True;regime:str="UNSPECIFIED";reason:str="";model_version:str="strategy"

class DashboardControlSchema(BaseModel):
    model_config=ConfigDict(frozen=True)
    mode:str;revision:int;updated_at:datetime;actor:str;reason:str;new_risk_enabled:bool

class DashboardCycleSchema(BaseModel):
    model_config=ConfigDict(frozen=True)
    cycle_id:str;timestamp:datetime;mark_price:Decimal;equity:Decimal;available_balance:Decimal
    gross_notional:Decimal;net_quantity:Decimal;target_net_quantity:Decimal;net_gap_quantity:Decimal
    long_quantity:Decimal;short_quantity:Decimal;long_target_quantity:Decimal;short_target_quantity:Decimal
    long_average_price:Decimal;short_average_price:Decimal;unrealized_pnl:Decimal;realized_pnl:Decimal
    funding_pnl:Decimal;fees:Decimal;ideal_order_count:int;submit_order_count:int;cancel_order_count:int
    fill_count:int;active_order_count:int;risk_blocked:bool;diagnostics:tuple[str,...];strategy:DashboardStrategySchema

class DashboardOverviewSchema(BaseModel):
    model_config=ConfigDict(frozen=True)
    account_id:str;symbol:str;mode:str;readiness_state:Literal["READY","NOT_READY","UNAVAILABLE"]
    readiness_reasons:tuple[str,...]=();control:DashboardControlSchema|None=None;latest:DashboardCycleSchema|None=None
    active_orders:tuple[dict[str,Any],...]=();telemetry_error:str|None=None
    operations:dict[str,Any]|None=None;operations_error:str|None=None;generated_at:datetime

class DashboardCyclesSchema(BaseModel):
    model_config=ConfigDict(frozen=True)
    cycles:tuple[DashboardCycleSchema,...];count:int

class DashboardDefaultsSchema(BaseModel):
    model_config=ConfigDict(frozen=True)
    account_id:str;symbol:str;refresh_seconds:int=5;local_only:bool=True

class DashboardControlRequest(BaseModel):
    action:Literal["pause","resume","reset"]
    reason:str=Field(default="",max_length=256)
