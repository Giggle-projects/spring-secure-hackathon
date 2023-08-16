package se.ton.t210.dto.ml;

import lombok.Getter;

@Getter
public class ExpectPassPercentResponse {

    private float prediction;

    public ExpectPassPercentResponse() {
    }

    public ExpectPassPercentResponse(int prediction) {
        this.prediction = prediction;
    }
}