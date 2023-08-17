package se.ton.t210.dto;

import lombok.Getter;

import java.time.LocalDate;
import java.util.Optional;

@Getter
public class AccessDateTimeFilter {

    private Optional<String> memberName;
    private Optional<LocalDate> dateFrom;
    private Optional<LocalDate> dateTo;

    public AccessDateTimeFilter(Optional<String> memberName, Optional<LocalDate> dateFrom, Optional<LocalDate> dateTo) {
        this.memberName = memberName;
        this.dateFrom = dateFrom;
        this.dateTo = dateTo;
    }
}
