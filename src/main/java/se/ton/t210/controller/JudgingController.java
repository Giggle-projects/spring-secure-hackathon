package se.ton.t210.controller;

import java.util.List;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.dto.JudgingItemNamesResponse;
import se.ton.t210.dto.TakenScoreResponse;
import se.ton.t210.service.JudgingService;

@RestController
public class JudgingController {

    private final JudgingService judgingService;

    public JudgingController(JudgingService judgingService) {
        this.judgingService = judgingService;
    }

    @GetMapping("/api/judge/score")
    public ResponseEntity<TakenScoreResponse> takenScore(Long judgingItemId, int memberScore) {
        final int takenScore = judgingService.calculateTakenScore(judgingItemId, memberScore);
        return ResponseEntity.ok(TakenScoreResponse.of(takenScore));
    }

    @GetMapping("/api/judge/names")
    public ResponseEntity<List<JudgingItemNamesResponse>> itemNames(ApplicationType applicationType) {
        final List<JudgingItemNamesResponse> names = judgingService.itemNames(applicationType);
        return ResponseEntity.ok(names);
    }
}
