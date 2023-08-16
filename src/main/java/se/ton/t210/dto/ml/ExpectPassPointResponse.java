package se.ton.t210.dto.ml;

import lombok.Getter;

@Getter
public class ExpectPassPointResponse {

    private int prediction;

    public ExpectPassPointResponse() {
    }

    public ExpectPassPointResponse(int prediction) {
        this.prediction = prediction;
    }
}
