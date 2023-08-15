package se.ton.t210.utils.date;

import java.time.LocalDate;
import java.time.Month;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class LocalDateUtils {

    public static List<LocalDate> monthsOfYear(LocalDate year) {
        return Arrays.stream(Month.values())
            .map(month -> LocalDate.of(year.getYear(), month, 1))
            .collect(Collectors.toList());
    }
}
