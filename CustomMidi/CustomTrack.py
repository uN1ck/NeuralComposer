class CustomTrack:
    """
    Класс отвечающий за конкретную дорожку в треке, хранит ее в разобранном виде c заданнйо частотой разбиения
    """

    def __init__(self, division: int):
        """
        Конструктор трека
        :param division: Частота разбиения трека
        """
        self.divisions = []
        self.division = division

    def get_data_set(self, step: int, sample_length: int, result_length: int = 1) -> [list, list]:
        """
        Метод доступа к подготовленному датасету, при каждом вызове формирует его заново. 
        :param step: Длина смещения в долях минимального разбиения
        :param sample_length: Длина единичного семпла для входных данных обучения
        :param result_length: Длина единичного семпла для выходных данных обучения, как правило равна 1
        :return: Пара массивов [X, y], где X входные, а y - выходные
        """
        in_divisions = []
        out_divisions = []

        for start in range(0, len(self.divisions) - sample_length, step):
            buffer = []
            for i in range(start, sample_length + start):
                buffer.append(self.divisions[i])
            in_divisions.append(buffer)
            out_divisions.append(self.divisions[sample_length + start])
        return [in_divisions, out_divisions]

    def get_segment_data_set(self, start: int, length: int) -> list:
        """
        Метод доступа к сегменту из датасета
        :param start: Индекс первой доли сегмента в датасете
        :param length: Длина сегмента
        :return: Массив долей заданной длины
        """
        # TODO: Ошибки на размер датасета?
        return self.divisions[start:start + length]
