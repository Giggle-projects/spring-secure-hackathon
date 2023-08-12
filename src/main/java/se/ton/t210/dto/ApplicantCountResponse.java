package se.ton.t210.dto;

import lombok.Getter;

@Getter
public class ApplicantCountResponse {

    private final int count;

    public ApplicantCountResponse(int count) {
        this.count = count;
    }
}
