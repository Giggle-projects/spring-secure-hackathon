package se.ton.t210.dto;

import lombok.Getter;

import java.util.List;

@Getter
public class EvaluationSectionsInfos {

    private final List<EvaluationSectionInfo> evaluationSectionInfos;

    public EvaluationSectionsInfos(List<EvaluationSectionInfo> evaluationSectionInfos) {
        this.evaluationSectionInfos = evaluationSectionInfos;
    }
}
