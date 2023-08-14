package se.ton.t210.dto;

import lombok.Getter;
import se.ton.t210.domain.type.ApplicationType;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@Getter
public class ApplicationTypeInfoResponse {

    private final ApplicationType applicationType;

    private final String applicationTypeStandardName;

    public ApplicationTypeInfoResponse(ApplicationType applicationType) {
        this.applicationType = applicationType;
        this.applicationTypeStandardName = applicationType.getStandardName();
    }

    public static List<ApplicationTypeInfoResponse> listOf() {
        return Arrays.stream(ApplicationType.values())
                .map(ApplicationTypeInfoResponse::new)
                .collect(Collectors.toList());
    }
}
