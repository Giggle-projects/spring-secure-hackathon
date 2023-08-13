package se.ton.t210.dto;

import lombok.Getter;

import java.util.List;

@Getter
public class ApplicationTypeNamesResponse {

    List<String> applicationTypeNames;

    public ApplicationTypeNamesResponse(List<String> applicationTypeNames) {
        this.applicationTypeNames = applicationTypeNames;
    }
}
