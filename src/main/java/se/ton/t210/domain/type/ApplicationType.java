package se.ton.t210.domain.type;

public enum ApplicationType {
    PoliceOfficerMale("/files/type_icon4.png", "경찰직공무원(남)", 1),
    PoliceOfficerFemale("/files/type_icon4.png", "경찰직공무원(여)", 2),
    FireOfficerMale("/files/type_icon2.jpeg", "소방직공무원(남)", 3),
    FireOfficerFemale("/files/type_icon2.jpeg", "소방직공무원(여)", 4),
    CorrectionalOfficerMale("/files/type_icon3.png", "경호직공무원(남)", 5),
    CorrectionalOfficerFemale("/files/type_icon3.png", "경호직공무원(여)", 6);

    private final String iconImageUrl;
    private final String standardName;
    private final int mlServerIndex;

    ApplicationType(String iconImageUrl, String standardName, int mlServerIndex) {
        this.iconImageUrl = iconImageUrl;
        this.standardName = standardName;
        this.mlServerIndex = mlServerIndex;
    }

    public String getStandardName() {
        return standardName;
    }

    public String iconImageUrl() {
        return iconImageUrl;
    }

    public int mlServerIndex() {
        return mlServerIndex;
    }
}
