package se.ton.t210.dto;

/*
지원자 현재 직렬 현재 점수 (이 달 평균 점수)
지원자 현재 직렬 최고 점수 (전체 최고 점수)
지원자 현재 직렬 종목별 현재 점수 (이 달 종목별 평균 점수)
지원자 현재 직렬 종목별 예상 점수
 */

import lombok.Getter;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Getter
public class MemberAverageScoresByJudgingItemResponse {

    private final List<MemberAverageScoresByJudgingItem> avgScoresByJudgingItems;

    public MemberAverageScoresByJudgingItemResponse(List<MemberAverageScoresByJudgingItem> avgScoresByJudgingItems) {
        this.avgScoresByJudgingItems = avgScoresByJudgingItems;
    }

    public static MemberAverageScoresByJudgingItemResponse listOf(Map<Long, Double> averageScoresByJudgingItem) {
        return new MemberAverageScoresByJudgingItemResponse(averageScoresByJudgingItem.keySet().stream()
            .map(it -> new MemberAverageScoresByJudgingItem(it, averageScoresByJudgingItem.get(it)))
            .collect(Collectors.toList())
        );
    }
}
