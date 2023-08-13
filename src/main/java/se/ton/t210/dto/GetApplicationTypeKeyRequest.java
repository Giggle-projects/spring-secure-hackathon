package se.ton.t210.dto;

import lombok.Getter;

@Getter
public class GetApplicationTypeKeyRequest {

    private String applicationTypeName;

    public GetApplicationTypeKeyRequest() {
    }

    public GetApplicationTypeKeyRequest(String applicationTypeName) {
        this.applicationTypeName = applicationTypeName;
    }
}
