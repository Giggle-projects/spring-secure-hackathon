package se.ton.t210.dto;

import lombok.Getter;
import se.ton.t210.domain.Member;
import se.ton.t210.domain.MemberScore;
import se.ton.t210.domain.type.ApplicationType;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

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

    public static RankResponse of(int rank, Member member, int score) {
        return new RankResponse(rank, member.getId(), member.getName(), member.getApplicationType(), score);
    }

    public static List<RankResponse> listOf(List<MemberScore> scores, Map<Long, Member> members) {
        final List<RankResponse> rankResponses = new ArrayList<>();
        int rank=0;
        for(MemberScore memberScore : scores) {
            final Member member = members.get(memberScore.getMemberId());
            final RankResponse rankResponse = RankResponse.of(rank, member, memberScore.getScore());
            rankResponses.add(rankResponse);
            rank++;
        }
        return rankResponses;
    }
}
