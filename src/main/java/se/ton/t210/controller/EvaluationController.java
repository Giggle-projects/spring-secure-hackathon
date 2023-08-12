package se.ton.t210.controller;

import java.util.List;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.dto.EvaluationItemNamesResponse;
import se.ton.t210.dto.TakenScoreResponse;
import se.ton.t210.service.EvaluationService;

@RestController
public class EvaluationController {

    private final EvaluationService evaluationService;

    public EvaluationController(EvaluationService evaluationService) {
        this.evaluationService = evaluationService;
    }

    @GetMapping("/api/judge/score")
    public ResponseEntity<TakenScoreResponse> takenScore(Long evaluationItemId, int score) {
        final int takenScore = evaluationService.calculateTakenScore(evaluationItemId, score);
        return ResponseEntity.ok(TakenScoreResponse.of(takenScore));
    }

    @GetMapping("/api/judge/names")
    public ResponseEntity<List<EvaluationItemNamesResponse>> itemNames(ApplicationType applicationType) {
        final List<EvaluationItemNamesResponse> names = evaluationService.itemNames(applicationType);
        return ResponseEntity.ok(names);
    }
}
