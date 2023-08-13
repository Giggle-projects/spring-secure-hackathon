package se.ton.t210.dto;

import lombok.Getter;

@Getter
public class GetApplicationTypeKeyResponse {

    private String applicationTypeKey;

    public GetApplicationTypeKeyResponse() {
    }

    public GetApplicationTypeKeyResponse(String applicationTypeKey) {
        this.applicationTypeKey = applicationTypeKey;
    }
}
