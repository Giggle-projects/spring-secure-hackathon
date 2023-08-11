package se.ton.t210.dto;

import java.util.List;
import java.util.stream.Collectors;
import lombok.Getter;
import se.ton.t210.domain.JudgingItem;

@Getter
public class JudgingItemNamesResponse {

    private final Long judgingItemId;

    private final String name;

    public JudgingItemNamesResponse(Long judgingItemId, String name) {
        this.judgingItemId = judgingItemId;
        this.name = name;
    }

    public static List<JudgingItemNamesResponse> listOf(List<JudgingItem> items) {
        return items.stream()
                .map(it -> new JudgingItemNamesResponse(it.getId(), it.getName()))
                .collect(Collectors.toList());
    }
}
