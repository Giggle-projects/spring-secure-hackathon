package se.ton.t210.dto;

import java.util.List;
import java.util.stream.Collectors;
import lombok.Getter;
import se.ton.t210.domain.EvaluationItem;

@Getter
public class EvaluationItemNamesResponse {

    private final Long evaluationItemId;

    private final String name;

    public EvaluationItemNamesResponse(Long evaluationItemId, String name) {
        this.evaluationItemId = evaluationItemId;
        this.name = name;
    }

    public static List<EvaluationItemNamesResponse> listOf(List<EvaluationItem> items) {
        return items.stream()
                .map(it -> new EvaluationItemNamesResponse(it.getId(), it.getName()))
                .collect(Collectors.toList());
    }
}
