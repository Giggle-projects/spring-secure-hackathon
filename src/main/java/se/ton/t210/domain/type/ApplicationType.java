package se.ton.t210.domain.type;

public enum ApplicationType {
    PoliceOfficerMale("/static/default.jpg", "경찰직공무원(남)", 1),
    PoliceOfficerFemale("/static/default.jpg", "경찰직공무원(여)", 2),
    FireOfficerMale("/static/default.jpg", "소방직공무원(남)", 3),
    FireOfficerFemale("/static/default.jpg", "소방직공무원(여)", 4),
    CorrectionalOfficerMale("/static/default.jpg", "경호직공무원(남)", 5),
    CorrectionalOfficerFemale("/static/default.jpg", "경호직공무원(여)", 6);

    private final String iconImageUrl;
    private final String standardName;
    private final int mlServerIndex;

    ApplicationType(String iconImageUrl, String standardName, int mlServerIndex) {
        this.iconImageUrl = iconImageUrl;
        this.standardName = standardName;
        this.mlServerIndex = mlServerIndex;
    }

    public String standardName() {
        return standardName;
    }

    public String iconImageUrl() {
        return iconImageUrl;
    }

    public int mlServerIndex() {
        return mlServerIndex;
    }
}
