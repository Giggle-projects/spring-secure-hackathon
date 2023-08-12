package se.ton.t210.dto;

import java.util.List;
import java.util.stream.Collectors;
import lombok.Getter;
import se.ton.t210.domain.EvaluationItem;

@Getter
public class EvaluationItemResponse {

    private final Long evaluationItemId;
    private final String evaluationName;

    public EvaluationItemResponse(Long evaluationItemId, String evaluationName) {
        this.evaluationItemId = evaluationItemId;
        this.evaluationName = evaluationName;
    }

    public static List<EvaluationItemResponse> listOf(List<EvaluationItem> items) {
        return items.stream()
                .map(it -> new EvaluationItemResponse(it.getId(), it.getName()))
                .collect(Collectors.toList());
    }
}
