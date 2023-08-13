package se.ton.t210.dto;

import lombok.Getter;
import se.ton.t210.domain.Member;
import se.ton.t210.domain.MonthlyScore;
import se.ton.t210.domain.type.ApplicationType;

@Getter
public class RankResponse {

    private final int rank;
    private final Long memberId;
    private final String memberName;
    private final ApplicationType applicationType;
    private final int score;

    public RankResponse(int rank, Long memberId, String memberName, ApplicationType applicationType, int score) {
        this.rank = rank;
        this.memberId = memberId;
        this.memberName = memberName;
        this.applicationType = applicationType;
        this.score = score;
    }

    public static RankResponse of(int rank, Member member, MonthlyScore score) {
        return new RankResponse(rank, member.getId(), member.getName(), member.getApplicationType(), score.getScore());
    }
}
