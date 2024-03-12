from numbers import Number


class LinearScheduler:
    """
    Parse anneal argument

    Usage:
    `v1@n1-v2@n2`: linearly increase value from v1 at update n1 to v2 at update n2
    """

    def __init__(self, params):
        self.schedule = self.parse_schedule(params)

    @staticmethod
    def parse_schedule(params):
        if params is None:
            return None

        if isinstance(params, Number):
            return [(params, 0)]

        assert isinstance(params, str)

        res = []
        last_update = -1
        for group in params.strip().split("-"):
            value, update = group.split("@")
            value = float(value)
            update = int(update.lower().replace("k", "000"))

            assert update > last_update
            res.append((value, update))
            last_update = update

        return res

    def get_value(self, update):
        if self.schedule is None:
            return None

        init_value, init_update = self.schedule[0]
        if update < init_update:
            return None

        for final_value, final_update in self.schedule[1:]:
            if update < final_update:
                return init_value + (final_value - init_value) * (update - init_update) / (final_update - init_update)

            init_value, init_update = final_value, final_update

        return init_value
