# -*- coding: utf-8 -*-
"""
This module contains everything used to extract the relevant data entries from
the sqlite database and store them in appropriately formated files. 

The input/output paths below need to be set acordingly. 

"""
from __future__ import annotations
from os import path, listdir
from datetime import datetime
import dataclasses
from enum import IntEnum, StrEnum
from typing import Sequence, Any
import re
import csv
import abc
import json
import ast
import numpy as np
import numpy.typing as npt

MOUSE_EVENTS_FOLDER = "mouse_events"
MOUSE_EVENTS_FILE_NAME = r'mm_([0-9]+)_([0-9]+)\.csv'
KEYBOARD_EVENTS_FOLDER = "keyboard_events"
KEYBOARD_EVENTS_FILE_NAME = r'ke_([0-9]+)_([0-9]+)\.csv'



def load_data(data_root: str) -> tuple[Sequence[Sentence],Sequence[Session],Sequence[User],Sequence[TaskAnswer]]:
    """Loads all interaction data from the respective files.
    
    Args:
        data_root (str): The path to the root directory containing the
            GUIDAETA dataset.
    Returns:
        tuple[Sequence[Sentence],Sequence[Session],Sequence[User],Sequence[TaskAnswer]]:
            The sequences of sentences, sessions, users and tasks.
    """
    assert path.isdir(data_root)
    sentences = Sentence.load_from_file(path.join(data_root, Sentence.SENTENCES_FILE_NAME))
    sessions = Session.load_from_file(path.join(data_root, Session.SESSIONS_FILE_NAME))
    task_answers = TaskAnswer.load_from_file(path.join(data_root, TaskAnswer.TASKS_FILE_NAME))
    users = User.load_from_file(path.join(data_root, User.USERS_FILE_NAME))
    for u in users:
        u.task_answers = [t for t in task_answers if t.user_id==u.id]
        for i,ta in enumerate(sorted(u.task_answers, key=lambda ta: ta.to_ts)):
            ta.sequence_order = i
    for t in task_answers:
        t.sessions = [s for s in sessions if s.task_answer_id==t.id]

    m_events_path = path.join(data_root, MOUSE_EVENTS_FOLDER)
    assert path.isdir(m_events_path)
    for file in listdir(m_events_path):
        m = re.match(MOUSE_EVENTS_FILE_NAME, file)
        if not m:
            print(f"[-] Unrecognized file name '{file}'")
            continue

        m_events = MouseEvent.load_from_file(path.join(m_events_path, file))
        for session_id in set(me.session_id for me in m_events):
            session = next((s for s in sessions if s.id==session_id), None)
            assert session
            assert not session.mouse_events
            session.mouse_events = [me for me in m_events if me.session_id==session_id]


    k_events_path = path.join(data_root, KEYBOARD_EVENTS_FOLDER)
    assert path.isdir(k_events_path)
    for file in listdir(k_events_path):
        m = re.match(KEYBOARD_EVENTS_FILE_NAME, file)
        if not m:
            print("[-] Unrecognized file name")
            print(file)
            continue

        k_events = KeyboardEvent.load_from_file(path.join(k_events_path, file))
        for session_id in set(ke.session_id for ke in k_events):
            session = next((s for s in sessions if s.id==session_id), None)
            assert session
            assert not session.keyboard_events
            session.keyboard_events = [ke for ke in k_events if ke.session_id==session_id]

    return sentences, sessions, users, task_answers


class Coord(IntEnum):
    """Int Enum for carthesian coordinates used in multidimensinal arrays."""
    X = 0
    Y = 1


class FromDataFile(abc.ABC):
    """Base class of dataclasses directly associated with a data file."""

    @property
    @abc.abstractmethod
    def data_field(self) -> type[IntEnum]:
        """Returns the int enum relating to the columns as they appear 
        in the respective data file.

        Returns:
            type[IntEnum]: The enumeration of the data fields.
        """

    @staticmethod
    @abc.abstractmethod
    def load_from_file(file_path: str) -> Sequence[FromDataFile]:
        """Loads the data from the given file and initializes a 
        sequence of respective class objects.

        Fails if the given path does not exist or is not a file.
        
        Args:
            file_path (str): The path to the file to load from.
        
        Returns:
            Sequence[FromDataFile]: Sequence of class objects created 
                from the data given in the provided file.
        """

    @staticmethod
    def load_data_records(file_path: str) -> list[list[str]]:
        """Loads the data from the given CSV file.

        Fails if the given path does not exist or is not a file.
        
        Args:
            file_path (str): The path to the file to load from.
        
        Returns:
            list[list[str]]: List of list of strings defining the 2D
                content of the CSV file.
        """
        assert path.isfile(file_path)
        with open(file_path, "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            return list(reader)


@dataclasses.dataclass
class Sentence(FromDataFile):
    """Data class for holding the sentences."""

    SENTENCES_FILE_NAME = "sentences.csv"

    class DataField(IntEnum):
        """Int enum relating to the columns as they appear in the 
        respective data file.
        """
        SENTENCE_ID = 0
        CHAPTER = 1
        CONTENT = 2

    @property
    def id(self) -> int:
        return self._id

    @property
    def data_field(self) -> type[DataField]:
        return Sentence.DataField

    @property
    def main_chapter(self) -> int:
        return int(self._chapter.split('.')[0])

    def __init__(self, data: list[str]):
        self._id = int(data[self.data_field.SENTENCE_ID])
        self._chapter = data[self.data_field.CHAPTER]
        self._content = data[self.data_field.CONTENT]

    @staticmethod
    def load_from_file(file_path: str) -> Sequence[Sentence]:
        return [Sentence(record) for record in __class__.load_data_records(file_path)]

@dataclasses.dataclass
class User(FromDataFile):
    """Data class for holding the users."""

    USERS_FILE_NAME = "users.csv"

    class DataField(IntEnum):
        """Int enum relating to the columns as they appear in the 
        respective data file.
        """
        USER_ID = 0
        DATE_JOINED = 1
        AGE = 2
        GENDER = 3
        EDUCATION = 4
        HANDEDNESS = 5
        AMBLYOPIA = 6
        DKT2_ANSWER_0 = 7
        DKT2_ANSWER_1 = 8
        DKT2_ANSWER_2 = 9
        DKT2_ANSWER_3 = 10
        DKT2_ANSWER_4 = 11
        DKT2_ANSWER_5 = 12
        DKT2_ANSWER_6 = 13
        DKT2_ANSWER_7 = 14
        DKT2_ANSWER_8 = 15
        DKT2_ANSWER_9 = 16
        DKT2_ANSWER_10 = 17
        DKT2_ANSWER_11 = 18
        DKT2_ANSWER_12 = 19
        DKT2_ANSWER_13 = 20

    class Gender(StrEnum):
        NA = "na"
        MALE = "m"
        FEMALE = "f"
        DIVERSE = "d"

    class Education(StrEnum):
        NA = "na"
        COMPULSARY_SCHOOL = "compulsoryschool"
        APPRENTICESHIP = "apprenticeship"
        HIGHSCHOOL = "highschool"
        UNIVERSITY = "university"

    class Handedness(StrEnum):
        LEFT = "left"
        RIGHT = "right"
        BOTH = "both"

    class Amblyopia(StrEnum):
        NA = "na"
        YES = "yes"

    class DiabetesKnowledge:

        REFERENCE_ANSWERS = ('b', 'c', 'a', 'd', 'c', 'b', 'b', 'c', 'a', 'b', 'a', 'c', 'b', 'd')
        def __init__(self, answers: tuple[str, ...]):
            assert len(answers)==len(__class__.REFERENCE_ANSWERS)
            self._answers = answers

        def score(self) -> float:
            return sum(np.array(self._answers)==np.array(__class__.REFERENCE_ANSWERS))/len(self._answers)

    @property
    def data_field(self) -> type[DataField]:
        return User.DataField

    @property
    def id(self) -> int:
        return self._id

    @property
    def age(self) -> int|None:
        return self._age

    @property
    def gender(self) -> Gender|None:
        return self._gender

    @property
    def education(self) -> Education|None:
        return self._education

    @property
    def handedness(self) -> Handedness|None:
        return self._handedness

    @property
    def amblyopia(self) -> Amblyopia|None:
        return self._amblyopia

    @property
    def diabetes_knowledge(self) -> DiabetesKnowledge:
        return self._diabetes_knowledge

    @property
    def task_answers(self) -> list[TaskAnswer]:
        return self._task_answers

    @task_answers.setter
    def task_answers(self, new_tasks: list[TaskAnswer]):
        self._task_answers = new_tasks

    def __init__(self, data: list[str]):
        self._id = int(data[self.data_field.USER_ID])
        self._joined = datetime.fromtimestamp(float(data[self.data_field.DATE_JOINED]))
        self._age = int(data[self.data_field.AGE]) if data[self.data_field.AGE] else None
        self._gender = User.Gender(data[self.data_field.GENDER]) if data[self.data_field.GENDER] else None
        self._education = User.Education(data[self.data_field.EDUCATION]) if data[self.data_field.EDUCATION] else None
        self._handedness = User.Handedness(data[self.data_field.HANDEDNESS]) if data[self.data_field.HANDEDNESS] else None
        self._amblyopia = User.Amblyopia(data[self.data_field.AMBLYOPIA]) if data[self.data_field.AMBLYOPIA] else None
        self._diabetes_knowledge = User.DiabetesKnowledge((
            data[self.data_field.DKT2_ANSWER_0],
            data[self.data_field.DKT2_ANSWER_1],
            data[self.data_field.DKT2_ANSWER_2],
            data[self.data_field.DKT2_ANSWER_3],
            data[self.data_field.DKT2_ANSWER_4],
            data[self.data_field.DKT2_ANSWER_5],
            data[self.data_field.DKT2_ANSWER_6],
            data[self.data_field.DKT2_ANSWER_7],
            data[self.data_field.DKT2_ANSWER_8],
            data[self.data_field.DKT2_ANSWER_9],
            data[self.data_field.DKT2_ANSWER_10],
            data[self.data_field.DKT2_ANSWER_11],
            data[self.data_field.DKT2_ANSWER_12],
            data[self.data_field.DKT2_ANSWER_13],
        ))
        self._task_answers: list[TaskAnswer] = []

    def mouse_events(self) -> list[MouseEvent]:
        """Returns all mouse events associated with this user.
        
        Returns:
            list[Session]: The list of the user's mouse events.

        """
        return [me for session in self.sessions() for me in session.mouse_events]

    def sessions(self) -> list[Session]:
        """Returns all sessions associated with this user.
        
        Returns:
            list[Session]: The list of the user's sessions.

        """
        return [s for t in self.task_answers for s in t.sessions]

    @staticmethod
    def load_from_file(file_path: str) -> Sequence[User]:
        return [User(record) for record in __class__.load_data_records(file_path)]


@dataclasses.dataclass
class TaskAnswer(FromDataFile):
    """Data class for holding the tasks."""

    TASKS_FILE_NAME = "task_answers.csv"
    DEFAULT_SEQUENCE_ORDER = -1
    NUM_CL_ANSWERS = 8

    class CognitiveLoad:

        @property
        def scores(self) -> npt.NDArray[np.int32]:
            return self._scores

        def __init__(self, scores: npt.NDArray[np.int32]):
            self._scores = scores
            assert np.all(self._scores==-1) or np.all((0 <= self._scores) & (self._scores <= TaskAnswer.NUM_CL_ANSWERS))

    class DataField(IntEnum):
        """Int enum relating to the columns as they appear in the 
        respective data file.
        """
        TASK_ANSWER_ID = 0
        USER_ID = 1
        TASK_NUMBER = 2
        TO_TS = 3
        CORRECTNESS = 4
        CL_ANSWER_0 = 5
        CL_ANSWER_1 = 6
        CL_ANSWER_2 = 7
        CL_ANSWER_3 = 8
        CL_ANSWER_4 = 9
        CL_ANSWER_5 = 10
        CL_ANSWER_6 = 11
        CL_ANSWER_7 = 12
        CL_ANSWER_8 = 13
        CL_ANSWER_9 = 14

    @property
    def data_field(self) -> type[DataField]:
        return TaskAnswer.DataField

    @property
    def id(self) -> int:
        return self._id

    @property
    def user_id(self) -> int:
        return self._user_id

    @property
    def task_no(self) -> int:
        return self._task_no

    @property
    def sequence_order(self) -> int:
        return self._sequence_order

    @property
    def to_ts(self) -> datetime:
        return self._to_ts

    @property
    def correctness(self) -> float:
        return self._correctness

    @property
    def sessions(self) -> list[Session]:
        return self._sessions

    @property
    def cl(self) -> CognitiveLoad:
        return self._cl

    @sessions.setter
    def sessions(self, new_sessions: list[Session]):
        self._sessions = new_sessions

    @sequence_order.setter
    def sequence_order(self, new_sequence_order: int):
        self._sequence_order = new_sequence_order

    def __init__(self, data: list[str]):
        self._id = int(data[self.data_field.TASK_ANSWER_ID])
        self._user_id = int(data[self.data_field.USER_ID])
        self._task_no = int(data[self.data_field.TASK_NUMBER])
        self._to_ts = datetime.fromtimestamp(float(data[self.data_field.TO_TS]))
        self._correctness = float(data[self.data_field.CORRECTNESS])
        assert 0 <= self._correctness <= 1.0
        self._correctness = float(data[self.data_field.CORRECTNESS])
        self._cl = TaskAnswer.CognitiveLoad(np.array([
            int(data[self.data_field.CL_ANSWER_0]),
            int(data[self.data_field.CL_ANSWER_1]),
            int(data[self.data_field.CL_ANSWER_2]),
            int(data[self.data_field.CL_ANSWER_3]),
            int(data[self.data_field.CL_ANSWER_4]),
            int(data[self.data_field.CL_ANSWER_5]),
            int(data[self.data_field.CL_ANSWER_6]),
            int(data[self.data_field.CL_ANSWER_7]),
            int(data[self.data_field.CL_ANSWER_8]),
            int(data[self.data_field.CL_ANSWER_9]),
        ]))
        self._sessions: list[Session] = []
        self._sequence_order: int = TaskAnswer.DEFAULT_SEQUENCE_ORDER

    def mouse_events(self) -> list[MouseEvent]:
        return [me for session in self._sessions for me in session.mouse_events]

    def keyboard_events(self) -> list[KeyboardEvent]:
        return [ke for session in self._sessions for ke in session.keyboard_events]

    @staticmethod
    def load_from_file(file_path: str) -> Sequence[TaskAnswer]:
        return [TaskAnswer(record) for record in __class__.load_data_records(file_path)]


@dataclasses.dataclass
class BaseEvent(FromDataFile):

    @property
    def session_id(self) -> int:
        return self._session_id

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    def __init__(self, session_id: int, timestamp: datetime):
        self._session_id = session_id
        self._timestamp = timestamp

    def _interpret_context(self, json_string: str) -> dict[str,Any]:
        if not json_string:
            return {}
        return ast.literal_eval(json_string)

@dataclasses.dataclass
class MouseEvent(BaseEvent):
    """Data class for holding the mouse events."""

    class DataField(IntEnum):
        """Int enum relating to the columns as they appear in the 
        respective data file.
        """
        SESSION_ID = 0
        TYPE = 1
        TIMESTAMP = 2
        ORIGIN_X = 3
        ORIGIN_Y = 4
        COMPONENT = 5
        CHAPTER = 6
        CONTEXT_TYPE = 7
        CONTEXT_DETAILS = 8

    class Type(IntEnum):
        """The different mouse event types."""
        MOVE = 0
        CLICK = 1
        SCROLL_UP = 2
        SCROLL_DOWN = 3

    class Component(IntEnum):
        """The different components used."""
        FULLTEXT = 0
        SNIPPETS = 1
        TILEBAR = 2
        WORDCLOUD = 3
        TOPICCLOUD = 4
        SEARCHBAR = 5
        IMAGESLIDER = 6

    class NormType(StrEnum):
        """Types of normalization for the screen-space coordinates.
        """
        INDEPENDENTLY = "independently"
        COMBINED = "combined"

    class Context:
        class Type(IntEnum):
            WC_TERM = 0
            """A click on a term in the WordCloud component."""
            TC_TERM = 1
            """ A click on a term in the TopicCloud component."""
            HC_TERM = 2
            """A click on a term in the HistoryWordCloud component."""
            CHAPTER_EXPAND = 3
            """A chapter is expanded."""
            CHAPTER_COLLAPSE = 4
            """A chapter is collapsed."""
            TOPICBAR = 5
            """A topic from a WordCloud's associated Topicbar is 
            activated or deactivated."""
            TB_ITEM = 6
            """A grid-element or row in the Tilebar component is clicked. 
            This can happend for the static display of the Tilebar at 
            the bottom of the Snippets."""
            SNIPPETS_EXPAND = 7
            """A text snippet in the Snippets component is expanded in 
            forward- or backwards direction via the respective expansion 
            button."""
            SNIPPETS_FULLTEXT = 8
            """If a header in the Snippets is clicked, the user jumps to 
            the respective location in the Fulltext component."""
            SEARCHBAR = 9
            """A click to the Searchbar component."""
            IMGS_NAVIGATE = 10
            """A click on the forward/backward arrows in the ImageSlider 
            component or the images modal."""
            IMGS_ENLARGE = 11
            """A click on an image in the ImageSlider which results in 
            an images modal with the respective image being displayed."""
            IMGS_CLOSE = 12
            """Closing the modal dialog showing an image which can be 
            triggered through clicking the close button or clicking 
            somewhere in the modal background."""
            CHANGE_VIS_MODE = 13
            """Chapter-wise toggle for changing the representation of a 
            chapter to either textual or visual format."""
            LIKE_BUTTON = 14
            """A click of the Like-Button in the Snippets component."""
            CHANGE_TEXT_ABSTRACTION = 15
            """Chapter-wise selection of the visual text abstraction 
            method, such as WordCloud or Topiccloud."""
            FULLTEXT_SENTENCE = 16
            """A click on a sentence in the fulltext view."""
            SNIPPETS_SENTENCE = 17
            """A click on a sentence in the snippets view."""

        def __init__(self, context_type: int, details: dict[str,Any]):
            self.type = __class__.Type(context_type)
            self.details: dict[str,Any] = details

        def __eq__(self, other: object) -> bool:
            return isinstance(other, MouseEvent.Context) \
                and self.type==other.type \
                and self.details==other.details

    @property
    def data_field(self) -> type[DataField]:
        return MouseEvent.DataField

    @property
    def type(self) -> MouseEvent.Type:
        return self._type

    @property
    def origin_x(self) -> int:
        return self._origin_x

    @property
    def origin_y(self) -> int:
        return self._origin_y

    @property
    def chapter(self) -> int|None:
        return self._chapter

    @property
    def component(self) -> MouseEvent.Component|None:
        return self._component

    @property
    def context(self) -> MouseEvent.Context|None:
        return self._context

    def __init__(self, data: list[str]):

        super().__init__(int(data[self.data_field.SESSION_ID]), datetime.fromtimestamp(float(data[self.data_field.TIMESTAMP])))
        self._type: MouseEvent.Type = MouseEvent.Type(int(data[self.data_field.TYPE]))
        self._origin_x: int = int(data[self.data_field.ORIGIN_X])
        self._origin_y: int = int(data[self.data_field.ORIGIN_Y])
        self._chapter: int|None = int(data[self.data_field.CHAPTER]) if data[self.data_field.CHAPTER] else None
        self._component: MouseEvent.Component|None = MouseEvent.Component(int(data[self.data_field.COMPONENT])) if data[self.data_field.COMPONENT] else None
        self._context: MouseEvent.Context|None = None
        if data[self.data_field.CONTEXT_TYPE]:
            self._context = MouseEvent.Context(
                int(data[self.data_field.CONTEXT_TYPE]),
                self._interpret_context(data[self.data_field.CONTEXT_DETAILS])
            )

    @staticmethod
    def load_from_file(file_path: str) -> list[MouseEvent]:
        """Loads all mouse events from the given data file path.

        Fails if the file is not found or not a file.

        Args:
            document_root (str): The path of the documents root.

        Returns:
            list[MouseEvent]: The list of mouse events.

        """
        return [MouseEvent(record) for record in __class__.load_data_records(file_path)]


@dataclasses.dataclass
class KeyboardEvent(BaseEvent):
    """Data class for holding the keyboard events."""

    class DataField(IntEnum):
        """Int enum relating to the columns as they appear in the 
        respective data file.
        """
        SESSION_ID = 0
        TIMESTAMP = 1
        KEY = 2
        MODIFIERS = 3
        SEARCH_STRING = 4

    @property
    def data_field(self) -> type[DataField]:
        return KeyboardEvent.DataField

    def __init__(self, data: list[str]):

        super().__init__(int(data[self.data_field.SESSION_ID]), datetime.fromtimestamp(float(data[self.data_field.TIMESTAMP])))
        self._key: str = data[self.data_field.KEY]
        self._modifiers: int = int(data[self.data_field.MODIFIERS])
        self._context: str = data[self.data_field.SEARCH_STRING]

    @staticmethod
    def load_from_file(file_path: str) -> list[KeyboardEvent]:
        """Loads all keyboard events from the given data file path.

        Fails if the file is not found or not a file.

        Args:
            document_root (str): The path of the documents root.

        Returns:
            list[KeyboardEvent]: The list of keyboard events.

        """
        return [KeyboardEvent(record) for record in __class__.load_data_records(file_path)]


@dataclasses.dataclass
class Session(FromDataFile):
    """Data class for holding the exploration sessions."""

    SESSIONS_FILE_NAME = "sessions.csv"

    class DataField(IntEnum):
        """Int enum relating to the columns as they appear in the 
        respective data file.
        """
        ID = 0
        TASK_ANSWER_ID = 1
        FROM_TS = 2
        TO_TS = 3
        INITIALIZATION_TYPE = 4
        FINALIZATION_TYPE = 5
        WINDOW_SIZES = 6

    class InitializationType(IntEnum):
        """Int enum for the different initialization types a session can have."""
        FIRST_VISIST = 0
        TAB_FOCUS = 1
        HELPERMODAL_CLOSE = 2
        TASKMODAL_CLOSE = 3
        USERMODAL_CLOSE = 4

    class FinalizationType(IntEnum):
        """Int enum for the different finalization types a session can have."""
        TAB_BLUR = 0
        HELPERMODAL_OPEN = 1
        TASKMODAL_OPEN = 2
        USERMODAL_OPEN = 3
        ON_EXPLORATION_DESTROY = 4
        INVALID = 5

    class WindowSize:

        class DataField(IntEnum):
            """Int enum relating to the columns as they appear in the 
            respective data file.
            """
            WIDTH = 0
            HEIGHT = 1

        def __init__(self, ts: datetime, width: int, height: int):
            self.ts = ts
            self.width = width
            self.height = height

    class InteractionInterval:
        def __init__(self, from_ts: datetime, to_ts: datetime):
            self.from_ts = from_ts
            self.to_ts = to_ts

        def duration(self) -> float:
            return (self.to_ts-self.from_ts).total_seconds()

    class Dwelling(InteractionInterval):
        """Unpertubed mouseover on a component for more prelong period 
        of time.
        """

        GRACE_PERIOD = 1.0
        """The grace period for pauses, time spent on other dwellings."""

        def __init__(self, component: MouseEvent.Component, chapter: int, from_ts: datetime):
            super().__init__(from_ts, from_ts)
            self.component = component
            self.chapter = chapter
            self.prev: Session.Dwelling|None = None
            self.next: Session.Dwelling|None = None

        def __eq__(self, other: object) -> bool:
            return isinstance(other, Session.Dwelling) \
                and self.component==other.component \
                and self.chapter==other.chapter

    class Hovering(InteractionInterval):
        """Hovering interval over an element.
        
        This differs from Dwelling as it is not the duration spent on a 
        specific comment, but derived from the mouse movement's hover
        context.
        """

        def __init__(self, component: MouseEvent.Component|None, chapter: int|None, context: MouseEvent.Context, from_ts: datetime):
            super().__init__(from_ts, from_ts)
            self.chapter = chapter
            self.component = component
            self.context = context

    @property
    def data_field(self) -> type[DataField]:
        return Session.DataField

    @property
    def id(self) -> int:
        return self._id

    @property
    def task_answer_id(self) -> int:
        return self._task_answer_id

    @property
    def from_ts(self) -> datetime:
        return self._from_ts

    @property
    def to_ts(self) -> datetime:
        return self._to_ts

    @property
    def finalization_type(self) -> int:
        return self._finalization_type

    @property
    def window_sizes(self) -> list[WindowSize]:
        return self._window_sizes

    @property
    def mouse_events(self) -> list[MouseEvent]:
        return self._mouse_events

    @mouse_events.setter
    def mouse_events(self, new_mouse_events: list[MouseEvent]):
        self._mouse_events = new_mouse_events

    @property
    def keyboard_events(self) -> list[KeyboardEvent]:
        return self._keyboard_events

    @keyboard_events.setter
    def keyboard_events(self, new_keyboard_events: list[KeyboardEvent]):
        self._keyboard_events = new_keyboard_events

    @property
    def events(self) -> Sequence[BaseEvent]:
        return sorted([*self._mouse_events, *self._keyboard_events], key=lambda e: e.timestamp)

    @property
    def dwellings(self) -> list[Session.Dwelling]:
        if self._dwellings:
            return self._dwellings
        return []

    @dwellings.setter
    def dwellings(self, new_dwellings: list[Session.Dwelling]):
        self._dwellings = new_dwellings

    @property
    def interactions(self) -> list[Any]:
        if self._interactions:
            return self._interactions
        return []

    @interactions.setter
    def interactions(self, new_interactions: list[Any]):
        self._interactions = new_interactions

    @property
    def hoverings(self) -> Sequence[Session.Hovering]:
        if self._hoverings:
            return self._hoverings
        return []

    @hoverings.setter
    def hoverings(self, new_hoverings: list[Session.Hovering]):
        self._hoverings = new_hoverings

    def __init__(self, data: list[str]):
        self._id = int(data[self.data_field.ID])
        self._task_answer_id = int(data[self.data_field.TASK_ANSWER_ID])
        self._from_ts = datetime.fromtimestamp(float(data[self.data_field.FROM_TS]))
        self._to_ts: datetime = datetime.fromtimestamp(float(data[self.data_field.TO_TS]))
        self._initialization_type = Session.InitializationType(int(data[self.data_field.INITIALIZATION_TYPE]))
        self._finalization_type = Session.FinalizationType(int(data[self.data_field.FINALIZATION_TYPE]))
        self._window_sizes = [
            Session.WindowSize(
                datetime.fromtimestamp(float(k)),
                v[Session.WindowSize.DataField.WIDTH],
                v[Session.WindowSize.DataField.HEIGHT])
            for k,v in json.loads(data[self.data_field.WINDOW_SIZES]).items()
        ]
        self._mouse_events = []
        self._keyboard_events = []
        self._dwellings: list[Session.Dwelling]|None = None
        self._hoverings: list[Session.Hovering]|None = None
        self._interactions: list[Any]|None = None

    def duration(self) -> float:
        """Returns the session's duration in seconds.
        
        Returns:
            float: Total seconds of the duration.
        """
        return (self._to_ts-self._from_ts).total_seconds()

    def get_normalized_positions(self, norm_type: MouseEvent.NormType = MouseEvent.NormType.INDEPENDENTLY) -> npt.NDArray[np.float64]:
        """Compute the normalized positions for the session. 

        As the window size can change over the course of the session, 
        the normalization is conducted for the intervals, specified 
        through the timestamps. 
        Note that normalized positions still can slightly exceed [0,1),
        for some events as the reported window sizes can be slightly out
        of sync when resizing a window.
        Returns an empy array if the session has no mouse events.

        Args:
            norm_type (MouseEvent.NormType): The normalization type.
        
        Returns:
            npt.NDArray[np.float64]: The array of normalized coordinates.
        """

        if not self.mouse_events:
            return np.array([])

        assert len(self.mouse_events)
        mouse_move_events = [me for me in self.mouse_events if me.type==MouseEvent.Type.MOVE]
        num_events = len(mouse_move_events)
        events_ts = np.array([me.timestamp for me in mouse_move_events])
        norm_pos = np.array([[me.origin_x, me.origin_y] for me in mouse_move_events]).astype(float)
        ws_mask = np.full((num_events), fill_value=False)
        for i,ws in enumerate(self.window_sizes):
            lower_thresh = ws.ts
            upper_thresh = self.window_sizes[i+1].ts if len(self.window_sizes)>(i+1) else None
            mask = events_ts >= lower_thresh
            if upper_thresh:
                mask &= (events_ts < upper_thresh)
            assert not np.any(ws_mask & mask)
            ws_mask |= mask
            if not np.any(mask):
                continue
            match norm_type:
                case MouseEvent.NormType.INDEPENDENTLY:
                    norm_pos[mask,Coord.X] /= ws.width
                    norm_pos[mask,Coord.Y] /= ws.height
                case MouseEvent.NormType.COMBINED:
                    norm_pos[mask] /= max(ws.width, ws.height)
                case _:
                    assert False
        assert np.all(ws_mask)
        assert np.all(norm_pos>=0) and np.all(norm_pos<=1.5)
        return norm_pos


    class SamplingScheme(StrEnum):
        """Different sampling schemes which can be eployed.
        
        Cursor position sampling is possible over either distance (with
        sampled points being evently spaced) or over time (sampled 
        points exhibiting uniform time intervals).
        The choice of sampling scheme depends on the intended usage.
        """
        DISTANCE = "distance"
        TIME = "time"

    def get_sampled_positions(self,
                              timestamps: npt.NDArray[np.float64],
                              positions: npt.NDArray[np.float64],
                              scheme: SamplingScheme=SamplingScheme.DISTANCE,
                              step: float=0.02,
                              gap_thresh: float=0.33
                              ) -> tuple[list[npt.NDArray[np.float64]],list[npt.NDArray[np.float64]]]:
        """Samples a sequence of positions over distance or time. 

        The given position sequence, identified by a list of timestamps
        and xy-coordinates is sampled using a distance- or time-based
        sampling scheme. 
        This involves three steps:
        1. Splitting the positions by gaps in the sequence. Due to 
            various reasons, there can be gaps in the sequence of 
            positions, with the cursor covering subtantial screen-space
            from one event to the next. A gap is identified through the 
            gap threshold.
        2. The resulting sequences are resampled according to the 
            specified scheme using the step threshold.
        3. As the sampling can result in very close-by points at sharp 
            turns, we apply a 2D filter removing all sequence points 
            exceeding a distance of sampling_thresh/2.
        
        Args:
            timestamps (npt.NDArray[np.float64]): The sequence of timestamps.
            positions (npt.NDArray[np.float64]): The sequence of 2D positions.
            scheme (SamplingScheme): The applied sampling scheme.
            step (float): The threshold for the sampling interval in 
                seconds for time-based and relative screenspace for 
                distance-based sampling.
            gap_thresh (float): The relativ screen space distance, 
                identifying a gap.

        Returns:
            tuple[list[npt.NDArray[np.float64]],list[npt.NDArray[np.float64]]]:
                The list of time sequences and list of distance sequences.
        """

        @dataclasses.dataclass
        class MoveSequence:
            """Dataclass containing an unperturbed mouse movement."""

            MIN_SEQUENCE_LENGTH = 10
            """Min number of events identifying a move sequence."""

            def __init__(self, timestamps: npt.NDArray[np.float64], positions: npt.NDArray[np.float64]):
                assert len(timestamps)==len(positions)
                self.timestamps = timestamps
                self.positions = positions

        def proximity_filter(sequence: MoveSequence, distance_thresh: float):
            xy = sequence.positions
            ts = sequence.timestamps
            while True:
                dists = np.sum(np.diff(xy, axis=0)**2, axis=1)**(1/2)
                too_close_points = np.where(dists < distance_thresh)[0]
                if not too_close_points.size:
                    break
                xy[too_close_points] = xy[too_close_points] + (xy[too_close_points+1]-xy[too_close_points])/2
                ts[too_close_points] = ts[too_close_points] + (ts[too_close_points+1]-ts[too_close_points])/2
                new_idxs = np.setdiff1d(np.array(range(len(xy))), too_close_points)
                xy = xy[new_idxs]
                ts = ts[new_idxs]
            return MoveSequence(ts,xy)

        def sample_by_distance(sequence: MoveSequence, step: float) -> None|MoveSequence:
            dists = np.sum(np.diff(sequence.positions, axis=0)**2, axis=1)**(1/2)
            total_dist = np.sum(dists)
            integral = np.concatenate(([0], np.cumsum(dists)))
            num_steps = int(total_dist / step)
            if num_steps < 2:
                return None

            sampled_points = np.empty((num_steps,2), dtype=float)
            sample_ts = np.empty((num_steps), dtype=float)
            for i in range(num_steps):
                lower_bound = sum(i*step >= integral)-1
                overhead = i*step - integral[lower_bound]
                relative_overhead = overhead/dists[lower_bound]
                assert 0 <= relative_overhead < 1
                sampled_points[i] = sequence.positions[lower_bound] + relative_overhead*(sequence.positions[lower_bound+1]-sequence.positions[lower_bound])
                sample_ts[i] = sequence.timestamps[lower_bound] + relative_overhead*(sequence.timestamps[lower_bound+1]-sequence.timestamps[lower_bound])
            return MoveSequence(sample_ts, sampled_points)

        def sample_by_time(sequence: MoveSequence, step: float) -> None|MoveSequence:
            sequence_duration = sequence.timestamps[-1]-sequence.timestamps[0]
            num_steps = int(sequence_duration / step)
            if num_steps < 2:
                return None
            sampled_points = np.empty((num_steps,2), dtype=float)
            sample_ts = np.empty((num_steps), dtype=float)
            offsets = sequence.timestamps-sequence.timestamps[0]
            for i in range(num_steps):
                lower_bound = sum(i*step >= offsets)-1
                overhead = i*step - offsets[lower_bound]
                assert overhead >= 0
                relative_overhead = overhead/(offsets[lower_bound+1]-offsets[lower_bound])
                assert 0 <= relative_overhead < 1
                sampled_points[i] = sequence.positions[lower_bound] + relative_overhead*(sequence.positions[lower_bound+1]-sequence.positions[lower_bound])
                sample_ts[i] = sequence.timestamps[lower_bound] + relative_overhead*(sequence.timestamps[lower_bound+1]-sequence.timestamps[lower_bound])
            return MoveSequence(sample_ts, sampled_points)

        # 1. Split positions by gaps in the sequence
        dists = np.sum(np.diff(positions, axis=0)**2, axis=1)**(1/2)
        cut_pts = np.where(dists>gap_thresh)[0]
        sequences: list[MoveSequence] = []
        start_idx = 0
        for cut_pt in cut_pts:
            if cut_pt-start_idx > MoveSequence.MIN_SEQUENCE_LENGTH:
                sequences.append(MoveSequence(timestamps[start_idx:cut_pt], positions[start_idx:cut_pt]))
            start_idx = cut_pt+1
        if len(timestamps)-start_idx > MoveSequence.MIN_SEQUENCE_LENGTH:
            sequences.append(MoveSequence(timestamps[start_idx:], positions[start_idx:]))


        sampled_sequences: list[MoveSequence] = []
        for sequence in sequences:
            # 2. Resample accoring to scheme
            match(scheme):
                case Session.SamplingScheme.DISTANCE:
                    sequence = sample_by_distance(sequence, step)
                case Session.SamplingScheme.TIME:
                    sequence = sample_by_time(sequence, step)
                case _:
                    assert False
            if not sequence:
                continue

            # 3. Filter out close-by points
            sampled_sequences += [proximity_filter(sequence, step/2)]

        return ([s.timestamps for s in sampled_sequences], [s.positions for s in sampled_sequences])

    @staticmethod
    def load_from_file(file_path: str) -> Sequence[Session]:
        """Loads all sessions from the respective session file.


        Fails if the `document_root` is not a directory or if the 
        sessions file is not found within it. 

        Args:
            document_root (str): The path of the documents root.

        Returns:
            list[Session]: The list of exploration sessions.

        """
        return [Session(record) for record in __class__.load_data_records(file_path)]


    def compute_hoverings(self):
        """Computes all the session's hoverings.

        Unlike for dwellings, there is no min-duration as threshold for
        defining a hovering.
        """
        current_hovering = None
        hoverings: list[Session.Hovering] = []
        for me in [me for me in self.mouse_events if me.type==MouseEvent.Type.MOVE]:
            if me.context is not None:
                if current_hovering \
                        and current_hovering.context==me.context \
                        and current_hovering.chapter==me.chapter \
                        and current_hovering.component==me.component:
                    current_hovering.to_ts = me.timestamp
                else:
                    if current_hovering:
                        hoverings.append(current_hovering)
                    current_hovering = Session.Hovering(me.component, me.chapter, me.context, me.timestamp)
            else:
                current_hovering = None
        if current_hovering:
            hoverings.append(current_hovering)
        self.hoverings = hoverings

    def compute_dwellings(self):
        """Computes all the session's dwellings. 
        
        This is a multi-step process:
        1. All the session's dwellings are computed as doubly linked 
            list. Initially all time intervals with unchanged chapter
            and component information are considered.
        2. Merge adjacent common dwellings, which are interrupted by 
            another (too short) dwelling.
        3. Merge adjacent dwellings if their separating pause is too 
            short.
        4. Remove too short dwellings.

        """

        prev_dwelling = None
        current_dwelling = None
        for me in [me for me in self.mouse_events if me.type==MouseEvent.Type.MOVE]:
            if me.component is not None and me.chapter is not None:
                if current_dwelling:
                    if me.component==current_dwelling.component and me.chapter==current_dwelling.chapter:
                        current_dwelling.to_ts = me.timestamp
                    else:
                        new_dwelling = Session.Dwelling(me.component, me.chapter, me.timestamp)
                        new_dwelling.prev = current_dwelling
                        current_dwelling.next = new_dwelling
                        current_dwelling = new_dwelling
                else:
                    current_dwelling = Session.Dwelling(me.component, me.chapter, me.timestamp)
                    if prev_dwelling:
                        current_dwelling.prev = prev_dwelling
                        prev_dwelling.next = current_dwelling
            else:
                if current_dwelling:
                    prev_dwelling = current_dwelling
                    current_dwelling = None
        if current_dwelling:
            prev_dwelling = current_dwelling

        if not prev_dwelling:
            return

        first_dwelling = prev_dwelling
        while first_dwelling.prev:
            first_dwelling = first_dwelling.prev
        current_dwelling = first_dwelling
        while current_dwelling:
            current_dwelling = current_dwelling.next
        current_dwelling = first_dwelling
        while current_dwelling.next:
            if current_dwelling.prev \
                and current_dwelling.prev==current_dwelling.next \
                and current_dwelling.prev!=current_dwelling \
                and (current_dwelling.next.from_ts-current_dwelling.prev.to_ts).total_seconds() < Session.Dwelling.GRACE_PERIOD:
                current_dwelling.prev.next = current_dwelling.next
                current_dwelling.next.prev = current_dwelling.prev
            current_dwelling = current_dwelling.next


        current_dwelling = first_dwelling
        while current_dwelling.next:
            if current_dwelling==current_dwelling.next \
                    and (current_dwelling.next.from_ts-current_dwelling.to_ts).total_seconds() < Session.Dwelling.GRACE_PERIOD:
                if current_dwelling.next.next:
                    current_dwelling.next.next.prev = current_dwelling
                current_dwelling.to_ts = current_dwelling.next.to_ts
                current_dwelling.next = current_dwelling.next.next

                if current_dwelling.next:
                    current_dwelling = current_dwelling.next
            else:
                current_dwelling = current_dwelling.next

        current_dwelling = first_dwelling
        while current_dwelling.next:
            if (current_dwelling.to_ts-current_dwelling.from_ts).total_seconds() < Session.Dwelling.GRACE_PERIOD:
                if current_dwelling.prev:
                    current_dwelling.prev.next = current_dwelling.next
                if current_dwelling.next:
                    current_dwelling.next.prev = current_dwelling.prev
                if current_dwelling is first_dwelling:
                    first_dwelling = current_dwelling.next
            current_dwelling = current_dwelling.next

        dwellings: list[Session.Dwelling] = []
        current_dwelling = first_dwelling
        while current_dwelling:
            dwellings.append(current_dwelling)
            current_dwelling = current_dwelling.next
        self.dwellings = dwellings
