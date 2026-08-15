"""Persistent-port model for grouped multi-leg actions."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from threading import RLock
from typing import Protocol
from uuid import UUID

from freqtrade.hedge.contracts.types import PositionSide


class ActionGroupMemberState(StrEnum):
    PLANNED = "PLANNED"
    SUBMITTED = "SUBMITTED"
    SKIPPED_ALREADY_FLAT = "SKIPPED_ALREADY_FLAT"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class ActionGroupMember:
    position_side: PositionSide
    state: ActionGroupMemberState = ActionGroupMemberState.PLANNED
    intent_id: str | None = None
    client_order_id: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ActionGroupRecord:
    action_group_id: UUID
    action_type: str
    account_id: str
    symbol: str
    members: tuple[ActionGroupMember, ...]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def member(self, side: PositionSide) -> ActionGroupMember:
        for item in self.members:
            if item.position_side is side:
                return item
        raise KeyError(side)


class ActionGroupRepository(Protocol):
    def put(self, group: ActionGroupRecord) -> None: ...

    def get(self, action_group_id: UUID) -> ActionGroupRecord | None: ...

    def update_member(self, action_group_id: UUID, member: ActionGroupMember) -> ActionGroupRecord: ...


class InMemoryActionGroupRepository:
    def __init__(self) -> None:
        self._groups: dict[UUID, ActionGroupRecord] = {}
        self._lock = RLock()

    def put(self, group: ActionGroupRecord) -> None:
        if not isinstance(group, ActionGroupRecord):
            raise TypeError("group must be an ActionGroupRecord")
        with self._lock:
            existing = self._groups.get(group.action_group_id)
            if existing is not None and existing != group:
                raise ValueError("action group already exists with conflicting data")
            self._groups[group.action_group_id] = group

    def get(self, action_group_id: UUID) -> ActionGroupRecord | None:
        with self._lock:
            return self._groups.get(action_group_id)

    def update_member(self, action_group_id: UUID, member: ActionGroupMember) -> ActionGroupRecord:
        with self._lock:
            current = self._groups[action_group_id]
            members = tuple(
                member if item.position_side is member.position_side else item
                for item in current.members
            )
            if members == current.members and all(
                item.position_side is not member.position_side for item in current.members
            ):
                raise KeyError(member.position_side)
            updated = replace(current, members=members, updated_at=datetime.now(UTC))
            self._groups[action_group_id] = updated
            return updated
