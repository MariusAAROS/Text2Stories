from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class Need(BaseModel):
    need: str = Field(
        description="A need is a requirement that is necessary to be fulfilled by the system."
    )
    benefit: str = Field(
        description="The benefit of the need. Why is it important to fulfill this need?"
    )
    requirements: List[str] = Field(
        description="Requirements that are necessary to fulfill the need."
    )

class Needs(BaseModel):
    needs: List[Need] = Field(
        description="List of needs that are part of the project."
    )   

class UserStory(BaseModel):
    user_story: str = Field(
        description="A user story is a description of a feature from an end-user perspective."
    )
    acceptance_criteria: List[str] = Field(
        description="Acceptance criteria are the conditions that a software product must satisfy to be accepted by a user."
    )
    independent: Optional[bool] | None = Field(
        default=None,
        description="Reflects the INVEST Independent criteria. It should not depend on another user story to be completed."
                    "If it does, it can be considered independent if you can solve the dependency i.e. it is not a blocker."
    )
    negotiable: Optional[bool] | None = Field(
        default=None,
        description="Reflects the INVEST Negotiable criteria. It should be open to discussion and change."
    )
    valuable: Optional[bool] | None = Field(
        default=None,
        description="Reflects the INVEST Valuable criteria. It should deliver value to the user."
    )
    estimable: Optional[bool] | None = Field(
        default=None,
        description="Reflects the INVEST Estimable criteria. It should be possible to estimate the necessary work load"
                    "to complete the user story."
    )
    small: Optional[bool] | None = Field(
        default=None,
        description="Reflects the INVEST Small criteria. It should be small enough to be completed in a single sprint."
    )
    testable: Optional[bool] | None = Field(
        default=None,
        description="Reflects the INVEST Testable criteria. It should be possible to test the user story."
    )

    def to_str(self) -> str:
        return f"**User Story**: {self.user_story}\n**Acceptance Criteria**:\n" + \
            "\n".join(self.acceptance_criteria) + "\n**INVEST**:\n" + \
            "\n".join([f"{k}: {v}" for k, v in self.model_dump().items() if k != "user_story" and k != "acceptance_criteria"])

    def to_str_no_invest(self) -> str:
        return f"**User Story**: {self.user_story}\n**Acceptance Criteria**:\n" + "\n".join(self.acceptance_criteria)
    
    def to_str_desc_only(self) -> str:
        return f"**User Story**: {self.user_story}"

class FlatUserStory(BaseModel):
    user_story: str = Field(
        description="A user story is a description of a feature from an end-user perspective."
    )
    epic: str = Field(
        description="Large body of work that can be broken down into a number of smaller tasks (called user stories)."
    )
    acceptance_criteria: List[str] = Field(
        description="Acceptance criteria are the conditions that a software product must satisfy to be accepted by a user."
    )
    independent: Optional[bool] | None = Field(
        default=None,
        description="Reflects the INVEST Independent criteria. It should not depend on another user story to be completed."
                    "If it does, it can be considered independent if you can solve the dependency i.e. it is not a blocker."
    )
    negotiable: Optional[bool] | None = Field(
        default=None,
        description="Reflects the INVEST Negotiable criteria. It should be open to discussion and change."
    )
    valuable: Optional[bool] | None = Field(
        default=None,
        description="Reflects the INVEST Valuable criteria. It should deliver value to the user."
    )
    estimable: Optional[bool] | None = Field(
        default=None,
        description="Reflects the INVEST Estimable criteria. It should be possible to estimate the necessary work load"
                    "to complete the user story."
    )
    small: Optional[bool] | None = Field(
        default=None,
        description="Reflects the INVEST Small criteria. It should be small enough to be completed in a single sprint."
    )
    testable: Optional[bool] | None = Field(
        default=None,
        description="Reflects the INVEST Testable criteria. It should be possible to test the user story."
    )

class UserStories(BaseModel):
    user_stories: List[UserStory] = Field(
        description="List of user stories that are part of the project."
    )

class Epic(BaseModel):
    epic: str = Field(
        description="Large body of work that can be broken down into a number of smaller tasks (called user stories)."
    )
    user_stories: Optional[List[UserStory]] = Field(
        description="User stories that are part of the epic."
    )

class Epics(BaseModel):
    epics: List[Epic] = Field(
        description="List of epics that are part of the project."
    )

class Feedback(BaseModel):
    grade: Literal["sufficient", "unsufficient"] = Field(
        description="Decide if the elicited needs are sufficient or not to create the written user stories")
    feedback: str = Field(
        description="If the elicited needs are not sufficient, please provide a feedback to the needs that might be missing."
                    "Invite it to be as detailed as possible and to look into its available tools.")