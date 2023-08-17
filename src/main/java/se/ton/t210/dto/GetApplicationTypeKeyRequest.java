package se.ton.t210.dto;

import lombok.Getter;

import javax.validation.constraints.NotBlank;

@Getter
public class GetApplicationTypeKeyRequest {

    @NotBlank
    private String applicationTypeName;

    public GetApplicationTypeKeyRequest() {
    }

    public GetApplicationTypeKeyRequest(String applicationTypeName) {
        this.applicationTypeName = applicationTypeName;
    }
}
