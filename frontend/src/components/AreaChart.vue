<template>
  <apexchart
    type="area"
    :series="series"
    :options="options"
  ></apexchart>
</template>

<script>
import VueApexCharts from "vue3-apexcharts";

export default {
  components: {
	  apexchart: VueApexCharts
  },
  props: {
    series: {
      required: true
    },
    options: {
      default: {}
    }
  }
}
</script>