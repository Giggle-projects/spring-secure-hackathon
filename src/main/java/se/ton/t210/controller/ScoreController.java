package se.ton.t210.controller;

import com.fasterxml.jackson.core.JsonProcessingException;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import se.ton.t210.configuration.annotation.LoginMember;
import se.ton.t210.dto.*;
import se.ton.t210.service.ScoreService;

import javax.validation.Valid;
import javax.validation.constraints.NotNull;
import java.time.LocalDate;
import java.util.List;

@Validated
@RestController
public class ScoreController {

    private final ScoreService scoreService;

    public ScoreController(ScoreService scoreService) {
        this.scoreService = scoreService;
    }

    @GetMapping("/api/score/count")
    public ResponseEntity<ScoreCountResponse> scoreCount(@LoginMember LoginMemberInfo member) {
        final ScoreCountResponse countResponse = scoreService.count(member.getApplicationType());
        return ResponseEntity.ok(countResponse);
    }

    @GetMapping("/api/score/evaluate")
    public ResponseEntity<CalculateEvaluationScoreResponse> evaluationScore(
            @Valid @NotNull Long evaluationItemId,
            @Valid @NotNull Float score) {
        final int evaluationScore = scoreService.evaluate(evaluationItemId, score);
        return ResponseEntity.ok(new CalculateEvaluationScoreResponse(evaluationScore));
    }

    @GetMapping("/api/score/year")
    public ResponseEntity<List<MonthlyScoreResponse>> scores(@LoginMember LoginMemberInfo member) {
        final List<MonthlyScoreResponse> scoresYear = scoreService.scoresYear(member, LocalDate.now());
        return ResponseEntity.ok(scoresYear);
    }

    @GetMapping("/api/score/expect")
    public ResponseEntity<ExpectScoreResponse> expect(@LoginMember LoginMemberInfo member) throws JsonProcessingException {
//        final ExpectScoreResponse scoreResponse = scoreService.expect(member, LocalDate.now());
        final ExpectScoreResponse scoreResponse = new ExpectScoreResponse(10, 10, 10.0f);
        return ResponseEntity.ok(scoreResponse);
    }

    @GetMapping("/api/score/me")
    public ResponseEntity<MyScoreResponse> myScore(@LoginMember LoginMemberInfo member) {
        final MyScoreResponse scoreResponse = scoreService.myScores(member.getId());
        return ResponseEntity.ok(scoreResponse);
    }

    @PostMapping("/api/score/me")
    public ResponseEntity<ScoreResponse> updateScore(@LoginMember LoginMemberInfo member,
                                                     @RequestBody List<EvaluationScoreRequest> request) {
        final ScoreResponse scoreResponse = scoreService.update(member.getId(), request, LocalDate.now());
        return ResponseEntity.ok(scoreResponse);
    }

    @GetMapping("/api/score/detail/me")
    public ResponseEntity<List<EvaluationScoreByItemResponse>> evaluationScoresOfItem(@LoginMember LoginMemberInfo member) {
        final List<EvaluationScoreByItemResponse> scores = scoreService.evaluationScores(member.getId(), member.getApplicationType(), LocalDate.now());
        return ResponseEntity.ok(scores);
    }

    @GetMapping("/api/score/detail/top")
    public ResponseEntity<List<EvaluationScoreByItemResponse>> rankersEvaluationScoresOfItem(@LoginMember LoginMemberInfo member,
                                                                                             int percent) {
        final List<EvaluationScoreByItemResponse> scores = scoreService.avgEvaluationItemScoresTopOf(member.getApplicationType(), percent, LocalDate.now());
        return ResponseEntity.ok(scores);
    }

    @GetMapping("/api/score/rank")
    public ResponseEntity<List<RankResponse>> rank(@Valid @NotNull Integer rankCnt,
                                                   @LoginMember LoginMemberInfo member) {
        final List<RankResponse> rankResponses = scoreService.rank(member.getApplicationType(), rankCnt, LocalDate.now());
        return ResponseEntity.ok(rankResponses);
    }
}
