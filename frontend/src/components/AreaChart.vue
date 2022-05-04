<template>
  <apexchart
    type="area"
    :options="options"
    :series="series"
  ></apexchart>
</template>

<script>
import VueApexCharts from "vue3-apexcharts";

export default {
  components: {
	  apexchart: VueApexCharts
  },
  props: {
    options: {
      default: {}
    },
    series: {
      required: true
    }
  }
}
</script>